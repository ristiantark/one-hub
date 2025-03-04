package replicate

import (
	"encoding/json"
	"io"
	"net/http"
	"one-api/common"
	"one-api/common/config"
	"one-api/common/requester"
	"one-api/common/utils"
	"one-api/types"
	"strings"
)

type ReplicateStreamHandler struct {
	Usage     *types.Usage
	ModelName string
	ID        string
	Provider  *ReplicateProvider
}

func (p ReplicateProvider) CreateChatCompletion(request types.ChatCompletionRequest) (types.ChatCompletionResponse, *types.OpenAIErrorWithStatusCode) {
	url, err := p.GetSupportedAPIUri(config.RelayModeChatCompletions)
	if err != nil {
		return nil, err
	}
	// 获取请求地址
	fullRequestURL := p.GetFullRequestURL(url, request.Model)
	if fullRequestURL == "" {
		return nil, common.ErrorWrapper(nil, "invalid_recraft_config", http.StatusInternalServerError)
	}
	// 获取请求头
	headers := p.GetRequestHeaders()
	replicateRequest := convertFromChatOpenai(request)
	req, err := p.Requester.NewRequest(http.MethodPost, fullRequestURL, p.Requester.WithBody(replicateRequest), p.Requester.WithHeader(headers))
	if err != nil {
		return nil, common.ErrorWrapper(err, "new_request_failed", http.StatusInternalServerError)
	}
	replicateResponse := &ReplicateResponse[[]string]{}
	// 发送请求
	_, err = p.Requester.SendRequest(req, replicateResponse, false)
	if err != nil {
		return nil, err
	}
	replicateResponseResult, err := getPrediction(p, replicateResponse)
	if err != nil {
		return nil, common.ErrorWrapper(err, "prediction_failed", http.StatusInternalServerError)
	}
	return p.convertToChatOpenai(replicateResponseResult)
}

func convertFromChatOpenai(request types.ChatCompletionRequest) *ReplicateRequest[ReplicateChatRequest] {
	systemPrompt := ""
	prompt := ""
	var imageUrl string
	
	// 设置最小 MaxTokens 为 1024
	if request.MaxTokens == 0 && request.MaxCompletionTokens > 0 {
		request.MaxTokens = request.MaxCompletionTokens
	}
	// 确保最小 token 不小于 1024
	if request.MaxTokens < 1024 {
		request.MaxTokens = 1024
	}
	
	// 寻找第一张图片 URL (目前 Replicate 只支持一张图片输入)
	for _, msg := range request.Messages {
		if msg.Role == "user" { // 只检查用户消息中的图片
			openaiContent := msg.ParseContent()
			for _, content := range openaiContent {
				if content.Type == types.ContentTypeImageURL && imageUrl == "" {
					// 找到第一张图片后记录并停止搜索
					imageUrl = content.ImageURL.URL
					break
				}
			}
			if imageUrl != "" {
				break // 已找到图片，不再继续搜索
			}
		}
	}
	
	// 构建消息文本
	for _, msg := range request.Messages {
		if msg.Role == "system" {
			systemPrompt += msg.StringContent() + "\n"
			continue
		}
		prompt += msg.Role + ": \n"
		openaiContent := msg.ParseContent()
		for _, content := range openaiContent {
			if content.Type == types.ContentTypeText {
				prompt += content.Text
			}
			// 图片URL已经单独处理，这里不再添加到prompt中
		}
		prompt += "\n"
	}
	prompt += "assistant: \n"
	
	return &ReplicateRequest[ReplicateChatRequest]{
		Stream: request.Stream,
		Input: ReplicateChatRequest{
			TopP:             request.TopP,
			MaxTokens:        request.MaxTokens,
			MinTokens:        0,
			Temperature:      request.Temperature,
			SystemPrompt:     systemPrompt,
			Prompt:           prompt,
			PresencePenalty:  request.PresencePenalty,
			FrequencyPenalty: request.FrequencyPenalty,
			Image:            imageUrl,
		},
	}
}

func (p ReplicateProvider) convertToChatOpenai(response *ReplicateResponse[[]string]) (*types.ChatCompletionResponse, *types.OpenAIErrorWithStatusCode) {
	responseText := ""
	if response.Output != nil {
		for _, text := range response.Output {
			responseText += text
		}
	}
	choice := types.ChatCompletionChoice{
		Index: 0,
		Message: types.ChatCompletionMessage{
			Role:    types.ChatMessageRoleAssistant,
			Content: responseText,
		},
		FinishReason: types.FinishReasonStop,
	}
	openaiResponse := &types.ChatCompletionResponse{
		ID:      response.ID,
		Object:  "chat.completion",
		Created: utils.GetTimestamp(),
		Choices: []types.ChatCompletionChoice{choice},
		Model:   response.Model,
		Usage: &types.Usage{
			CompletionTokens: 0,
			PromptTokens:     0,
			TotalTokens:      0,
		},
	}
	p.Usage.PromptTokens = response.Metrics.InputTokenCount
	p.Usage.CompletionTokens = response.Metrics.OutputTokenCount
	p.Usage.TotalTokens = p.Usage.PromptTokens + p.Usage.CompletionTokens
	openaiResponse.Usage = p.Usage
	return openaiResponse, nil
}

func (p ReplicateProvider) CreateChatCompletionStream(request types.ChatCompletionRequest) (requester.StreamReaderInterface[string], *types.OpenAIErrorWithStatusCode) {
	url, err := p.GetSupportedAPIUri(config.RelayModeChatCompletions)
	if err != nil {
		return nil, err
	}
	// 获取请求地址
	fullRequestURL := p.GetFullRequestURL(url, request.Model)
	if fullRequestURL == "" {
		return nil, common.ErrorWrapper(nil, "invalid_recraft_config", http.StatusInternalServerError)
	}
	// 获取请求头
	headers := p.GetRequestHeaders()
	replicateRequest := convertFromChatOpenai(request)
	req, err := p.Requester.NewRequest(http.MethodPost, fullRequestURL, p.Requester.WithBody(replicateRequest), p.Requester.WithHeader(headers))
	if err != nil {
		return nil, common.ErrorWrapper(err, "new_request_failed", http.StatusInternalServerError)
	}
	replicateResponse := &ReplicateResponse[[]string]{}
	// 发送请求
	_, err = p.Requester.SendRequest(req, replicateResponse, false)
	if err != nil {
		return nil, err
	}
	headers["Accept"] = "text/event-stream"
	req, err = p.Requester.NewRequest(http.MethodGet, replicateResponse.Urls.Stream, p.Requester.WithHeader(headers))
	if err != nil {
		return nil, common.ErrorWrapper(err, "new_request_failed", http.StatusInternalServerError)
	}
	// 发送请求
	resp, err := p.Requester.SendRequestRaw(req)
	if err != nil {
		return nil, err
	}
	chatHandler := ReplicateStreamHandler{
		Usage:     p.Usage,
		ModelName: request.Model,
		ID:        replicateResponse.ID,
		Provider:  p,
	}
	return requester.RequestStream(p.Requester, resp, chatHandler.HandlerChatStream)
}

func (h ReplicateStreamHandler) HandlerChatStream(rawLine []byte, dataChan chan string, errChan chan error) {
	// 检查是否是完成事件
	if strings.HasPrefix(string(rawLine), "event: done") {
		// 获取用量
		replicateResponse := getPredictionResponse[[]string](h.Provider, h.ID)
		h.Usage.PromptTokens = replicateResponse.Metrics.InputTokenCount
		h.Usage.CompletionTokens = replicateResponse.Metrics.OutputTokenCount
		h.Usage.TotalTokens = h.Usage.PromptTokens + h.Usage.CompletionTokens
		// 需要有一个stop
		choice := types.ChatCompletionStreamChoice{
			Index: 0,
			Delta: types.ChatCompletionStreamChoiceDelta{
				Role: types.ChatMessageRoleAssistant,
			},
			FinishReason: types.FinishReasonStop,
		}
		dataChan <- getStreamResponse(h.ID, choice, h.ModelName)
		errChan <- io.EOF
		*rawLine = requester.StreamClosed
		return
	}
	
	// 如果rawLine 前缀不为data:，则直接返回
	if !strings.HasPrefix(string(rawLine), "data: ") {
		*rawLine = nil
		return
	}
	
	// 去除前缀 "data: "
	data := string(rawLine)[6:]
	
	// 如果数据为空（只有data:），表示是一个换行
	if len(strings.TrimSpace(data)) == 0 {
		// 发送换行符
		choice := types.ChatCompletionStreamChoice{
			Index: 0,
			Delta: types.ChatCompletionStreamChoiceDelta{
				Role:    types.ChatMessageRoleAssistant,
				Content: "\n\n", // 用两个换行符表示段落分隔
			},
		}
		dataChan <- getStreamResponse(h.ID, choice, h.ModelName)
	} else {
		// 发送常规文本内容
		choice := types.ChatCompletionStreamChoice{
			Index: 0,
			Delta: types.ChatCompletionStreamChoiceDelta{
				Role:    types.ChatMessageRoleAssistant,
				Content: data,
			},
		}
		dataChan <- getStreamResponse(h.ID, choice, h.ModelName)
	}
}

func getStreamResponse(id string, choice types.ChatCompletionStreamChoice, modelName string) string {
	chatCompletion := types.ChatCompletionStreamResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: utils.GetTimestamp(),
		Model:   modelName,
		Choices: []types.ChatCompletionStreamChoice{choice},
	}
	responseBody, _ := json.Marshal(chatCompletion)
	return string(responseBody)
}
