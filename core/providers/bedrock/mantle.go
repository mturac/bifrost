package bedrock

import (
	"bytes"
	"context"
	"fmt"
	"maps"
	"net/http"
	"strings"

	openai "github.com/maximhq/bifrost/core/providers/openai"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// isMantleModel reports whether a model should be routed via the Bedrock Mantle endpoint.
// Accepts "gpt-oss-120b", "openai.gpt-oss-120b", or region-prefixed variants.
func isMantleModel(model string) bool {
	return strings.Contains(model, "gpt-oss")
}

// mantleModelID returns the model identifier expected by the Mantle endpoint.
// e.g. "gpt-oss-120b" → "openai.gpt-oss-120b"; already-prefixed inputs are not double-prefixed.
func mantleModelID(model string) string {
	_, bareModel := parseBedrockRegionAndModel(model)
	bareModel = strings.TrimPrefix(bareModel, "openai.")
	if strings.HasPrefix(bareModel, "gpt-oss") {
		return "openai." + bareModel
	}
	return bareModel
}

// mantleURL builds the Bedrock Mantle endpoint URL for the given region and API path.
func mantleURL(region, path string) string {
	return fmt.Sprintf("https://bedrock-mantle.%s.api.aws/v1/%s", region, path)
}

// mantleSigV4Headers computes SigV4 auth headers for a mantle request by signing a dummy
// net/http.Request. jsonData must be the exact bytes that will be sent. accept must match
// the Accept header the actual request will send, since SigV4 signs all request headers.
func (provider *BedrockProvider) mantleSigV4Headers(
	ctx *schemas.BifrostContext,
	jsonData []byte,
	requestURL, accept string,
	key schemas.Key,
	region string,
) (map[string]string, *schemas.BifrostError) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(jsonData))
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError("failed to create signing request", err)
	}
	req.Header.Set("Accept", accept)
	if bifrostErr := signAWSRequestFromKey(ctx, req, key.BedrockKeyConfig, region, bedrockMantleSigningService); bifrostErr != nil {
		return nil, bifrostErr
	}
	headers := map[string]string{
		"Authorization":        req.Header.Get("Authorization"),
		"X-Amz-Date":           req.Header.Get("X-Amz-Date"),
		"x-amz-content-sha256": req.Header.Get("x-amz-content-sha256"),
		"Accept":               accept,
	}
	if token := req.Header.Get("X-Amz-Security-Token"); token != "" {
		headers["X-Amz-Security-Token"] = token
	}
	return headers, nil
}

// chatCompletionViaMantle handles non-streaming chat completions for mantle (gpt-oss) models.
func (provider *BedrockProvider) chatCompletionViaMantle(
	ctx *schemas.BifrostContext,
	key schemas.Key,
	request *schemas.BifrostChatRequest,
) (*schemas.BifrostChatResponse, *schemas.BifrostError) {
	region := resolveBedrockRegion(key, request.Model)

	reqCopy := *request
	reqCopy.Provider = schemas.OpenAI
	reqCopy.Model = mantleModelID(request.Model)

	url := mantleURL(region, "chat/completions")

	// Build extraHeaders: always start with network-config headers, then overlay SigV4 if needed.
	extraHeaders := maps.Clone(provider.networkConfig.ExtraHeaders)
	if key.Value.GetValue() == "" {
		// SigV4: pre-build body for signing. HandleOpenAIChatCompletionRequest rebuilds the
		// same bytes (deterministic marshaling), so the signature stays valid.
		jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(ctx, &reqCopy, func() (providerUtils.RequestBodyWithExtraParams, error) {
			return openai.ToOpenAIChatRequest(ctx, &reqCopy), nil
		})
		if bifrostErr != nil {
			return nil, bifrostErr
		}
		sigHeaders, bifrostErr := provider.mantleSigV4Headers(ctx, jsonData, url, "application/json", key, region)
		if bifrostErr != nil {
			return nil, bifrostErr
		}
		maps.Copy(extraHeaders, sigHeaders) // SigV4 headers override any conflicting network headers
	}

	response, bifrostErr := openai.HandleOpenAIChatCompletionRequest(
		ctx,
		provider.mantleClient,
		url,
		&reqCopy,
		key,
		extraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		schemas.Bedrock,
		nil, nil,
		provider.logger,
	)
	if response != nil {
		response.Model = request.Model
	}
	return response, bifrostErr
}

// chatCompletionStreamViaMantle handles streaming chat completions for mantle (gpt-oss) models.
func (provider *BedrockProvider) chatCompletionStreamViaMantle(
	ctx *schemas.BifrostContext,
	postHookRunner schemas.PostHookRunner,
	postHookSpanFinalizer func(context.Context),
	key schemas.Key,
	request *schemas.BifrostChatRequest,
) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	region := resolveBedrockRegion(key, request.Model)

	reqCopy := *request
	reqCopy.Provider = schemas.OpenAI
	reqCopy.Model = mantleModelID(request.Model)

	url := mantleURL(region, "chat/completions")
	originalModel := request.Model

	// Bearer: identical to Groq / any OpenAI-compatible provider.
	if key.Value.GetValue() != "" {
		authHeader := map[string]string{"Authorization": "Bearer " + key.Value.GetValue()}
		return openai.HandleOpenAIChatCompletionStreaming(
			ctx, provider.mantleStreamingClient, url, &reqCopy,
			authHeader, provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			schemas.Bedrock, postHookRunner,
			nil, nil, nil, nil,
			func(r *schemas.BifrostChatResponse) *schemas.BifrostChatResponse {
				if r != nil {
					r.Model = originalModel
				}
				return r
			},
			provider.logger, postHookSpanFinalizer,
		)
	}

	// SigV4: pre-build body to sign, then pass it via customRequestConverter so the handler
	// sends the exact same bytes we signed.
	openaiReq := openai.ToOpenAIChatRequest(ctx, &reqCopy)
	openaiReq.Stream = schemas.Ptr(true)
	openaiReq.StreamOptions = &schemas.ChatStreamOptions{IncludeUsage: schemas.Ptr(true)}

	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(ctx, &reqCopy, func() (providerUtils.RequestBodyWithExtraParams, error) {
		return openaiReq, nil
	})
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	authHeader, bifrostErr := provider.mantleSigV4Headers(ctx, jsonData, url, "text/event-stream", key, region)
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	return openai.HandleOpenAIChatCompletionStreaming(
		ctx, provider.mantleStreamingClient, url, &reqCopy,
		authHeader, provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		schemas.Bedrock, postHookRunner,
		func(_ *schemas.BifrostChatRequest) (providerUtils.RequestBodyWithExtraParams, error) {
			return openaiReq, nil
		},
		nil, nil, nil,
		func(r *schemas.BifrostChatResponse) *schemas.BifrostChatResponse {
			if r != nil {
				r.Model = originalModel
			}
			return r
		},
		provider.logger, postHookSpanFinalizer,
	)
}

// responsesViaMantle handles non-streaming Responses API requests for mantle (gpt-oss) models.
func (provider *BedrockProvider) responsesViaMantle(
	ctx *schemas.BifrostContext,
	key schemas.Key,
	request *schemas.BifrostResponsesRequest,
) (*schemas.BifrostResponsesResponse, *schemas.BifrostError) {
	region := resolveBedrockRegion(key, request.Model)

	reqCopy := *request
	reqCopy.Model = mantleModelID(request.Model)

	url := mantleURL(region, "responses")

	extraHeaders := maps.Clone(provider.networkConfig.ExtraHeaders)
	if key.Value.GetValue() == "" {
		jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(ctx, &reqCopy, func() (providerUtils.RequestBodyWithExtraParams, error) {
			return openai.ToOpenAIResponsesRequest(&reqCopy), nil
		})
		if bifrostErr != nil {
			return nil, bifrostErr
		}
		sigHeaders, bifrostErr := provider.mantleSigV4Headers(ctx, jsonData, url, "application/json", key, region)
		if bifrostErr != nil {
			return nil, bifrostErr
		}
		maps.Copy(extraHeaders, sigHeaders)
	}

	response, bifrostErr := openai.HandleOpenAIResponsesRequest(
		ctx,
		provider.mantleClient,
		url,
		&reqCopy,
		key,
		extraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		schemas.Bedrock,
		nil, nil,
		provider.logger,
	)
	if response != nil {
		response.Model = request.Model
	}
	return response, bifrostErr
}

// responsesStreamViaMantle handles streaming Responses API requests for mantle (gpt-oss) models.
func (provider *BedrockProvider) responsesStreamViaMantle(
	ctx *schemas.BifrostContext,
	postHookRunner schemas.PostHookRunner,
	postHookSpanFinalizer func(context.Context),
	key schemas.Key,
	request *schemas.BifrostResponsesRequest,
) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	region := resolveBedrockRegion(key, request.Model)

	reqCopy := *request
	reqCopy.Model = mantleModelID(request.Model)

	url := mantleURL(region, "responses")
	originalModel := request.Model

	// Bearer: identical to Groq / any OpenAI-compatible provider.
	if key.Value.GetValue() != "" {
		authHeader := map[string]string{"Authorization": "Bearer " + key.Value.GetValue()}
		return openai.HandleOpenAIResponsesStreaming(
			ctx, provider.mantleStreamingClient, url, &reqCopy,
			authHeader, provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			schemas.Bedrock, postHookRunner,
			nil, nil, nil,
			func(r *schemas.BifrostResponsesStreamResponse) *schemas.BifrostResponsesStreamResponse {
				if r != nil && r.Response != nil {
					r.Response.Model = originalModel
				}
				return r
			},
			provider.logger, postHookSpanFinalizer,
		)
	}

	// SigV4: pre-build body to sign.
	openaiReq := openai.ToOpenAIResponsesRequest(&reqCopy)
	openaiReq.Stream = schemas.Ptr(true)

	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(ctx, &reqCopy, func() (providerUtils.RequestBodyWithExtraParams, error) {
		return openaiReq, nil
	})
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	authHeader, bifrostErr := provider.mantleSigV4Headers(ctx, jsonData, url, "text/event-stream", key, region)
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	return openai.HandleOpenAIResponsesStreaming(
		ctx, provider.mantleStreamingClient, url, &reqCopy,
		authHeader, provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		schemas.Bedrock, postHookRunner,
		nil, nil,
		func(_ *openai.OpenAIResponsesRequest) *openai.OpenAIResponsesRequest {
			return openaiReq
		},
		func(r *schemas.BifrostResponsesStreamResponse) *schemas.BifrostResponsesStreamResponse {
			if r != nil && r.Response != nil {
				r.Response.Model = originalModel
			}
			return r
		},
		provider.logger, postHookSpanFinalizer,
	)
}
