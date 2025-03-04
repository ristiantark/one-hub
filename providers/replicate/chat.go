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
	"bytes"
)

type ReplicateStreamHandler struct {
	Usage       *types.Usage
	ModelName   string
	ID          string
	Provider    *ReplicateProvider
	Buffer      []string       // 用于缓存连续的空行
	EventBuffer bytes.Buffer   // 用于缓存整个事件
	IsEvent     bool           // 标记当前是否在处理事件
	CurrentEvent string        // 当前事件类型
}

func (p *ReplicateProvider) CreateChatCompletion(request *types.ChatCompletionRequest) (response *types.ChatCompletionResponse, errWithCode *types.OpenAIErrorWithStatusCode) {
	url, errWithCode := p.GetSupportedAPIUri(config.RelayModeChatCompletions)
	if errWithCode != nil {
		return nil, errWithCode
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
	_, errWithCode = p.Requester.SendRequest(req, replicateResponse, false)
	if errWithCode != nil {
		return nil, errWithCode
	}

	replicateResponse, err = getPrediction(p, replicateResponse)
	if err != nil {
		return nil, common.ErrorWrapper(err, "prediction_failed", http.StatusInternalServerError)
	}

	return p.convertToChatOpenai(replicateResponse)
}

func convertFromChatOpenai(request *types.ChatCompletionRequest) *ReplicateRequest[ReplicateChatRequest] {
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
			} else if content.Type == types.ContentTypeImageURL {
				// 处理图片URL - 使用最后一个图片
				imageUrl = content.ImageURL.URL
			}
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

func (p *ReplicateProvider) convertToChatOpenai(response *ReplicateResponse[[]string]) (*types.ChatCompletionResponse, *types.OpenAIErrorWithStatusCode) {
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

func (p *ReplicateProvider) CreateChatCompletionStream(request *types.ChatCompletionRequest) (requester.StreamReaderInterface[string], *types.OpenAIErrorWithStatusCode) {
	url, errWithCode := p.GetSupportedAPIUri(config.RelayModeChatCompletions)
	if errWithCode != nil {
		return nil, errWithCode
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
	_, errWithCode = p.Requester.SendRequest(req, replicateResponse, false)
	if errWithCode != nil {
		return nil, errWithCode
	}

	headers["Accept"] = "text/event-stream"
	req, err = p.Requester.NewRequest(http.MethodGet, replicateResponse.Urls.Stream, p.Requester.WithHeader(headers))

	if err != nil {
		return nil, common.ErrorWrapper(err, "new_request_failed", http.StatusInternalServerError)
	}

	// 发送请求
	resp, errWithCode := p.Requester.SendRequestRaw(req)
	if errWithCode != nil {
		return nil, errWithCode
	}

	chatHandler := ReplicateStreamHandler{
		Usage:     p.Usage,
		ModelName: request.Model,
		ID:        replicateResponse.ID,
		Provider:  p,
		Buffer:    make([]string, 0),
	}

	return requester.RequestStream(p.Requester, resp, chatHandler.HandlerChatStream)
}

func (h *ReplicateStreamHandler) HandlerChatStream(rawLine *[]byte, dataChan chan string, errChan chan error) {
	line := string(*rawLine)

	// 处理结束事件
	if strings.HasPrefix(line, "event: done") {
		// 获取用量
		replicateResponse := getPredictionResponse[[]string](h.Provider, h.ID)

		h.Usage.PromptTokens = replicateResponse.Metrics.InputTokenCount
		h.Usage.CompletionTokens = replicateResponse.Metrics.OutputTokenCount
		h.Usage.TotalTokens = h.Usage.PromptTokens + h.Usage.CompletionTokens

		// 如果缓冲区中还有数据，先发送
		if len(h.Buffer) > 0 {
			newlineCount := len(h.Buffer) - 1 // 空行数量转换为换行符数量
			if newlineCount > 0 {
				content := strings.Repeat("\n", newlineCount)
				choice := types.ChatCompletionStreamChoice{
					Index: 0,
					Delta: types.ChatCompletionStreamChoiceDelta{
						Role:    types.ChatMessageRoleAssistant,
						Content: content,
					},
				}
				dataChan <- getStreamResponse(h.ID, choice, h.ModelName)
			}
			h.Buffer = nil
		}

		// 发送结束信号
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

	// 检测事件开始
	if strings.HasPrefix(line, "event: ") {
		h.IsEvent = true
		h.CurrentEvent = strings.TrimPrefix(line, "event: ")
		h.EventBuffer.Reset()
		*rawLine = nil
		return
	}

	// 如果不是数据行且不是我们关心的事件，跳过
	if !strings.HasPrefix(line, "data: ") && !h.IsEvent {
		*rawLine = nil
		return
	}

	// 如果是数据行
	if strings.HasPrefix(line, "data: ") {
		// 只处理output事件的数据
		if h.CurrentEvent == "output" {
			content := strings.TrimPrefix(line, "data: ")
			
			// 处理空行
			if len(strings.TrimSpace(content)) == 0 {
				// 累积空行
				h.Buffer = append(h.Buffer, "")
				*rawLine = nil
				return
			} else {
				// 有实际内容，先处理之前积累的空行
				if len(h.Buffer) > 0 {
					newlineCount := len(h.Buffer) - 1 // 空行数量转换为换行符数量
					if newlineCount > 0 {
						newlineContent := strings.Repeat("\n", newlineCount)
						choice := types.ChatCompletionStreamChoice{
							Index: 0,
							Delta: types.ChatCompletionStreamChoiceDelta{
								Role:    types.ChatMessageRoleAssistant,
								Content: newlineContent,
							},
						}
						dataChan <- getStreamResponse(h.ID, choice, h.ModelName)
					}
					h.Buffer = nil
				}
				
				// 发送实际内容
				choice := types.ChatCompletionStreamChoice{
					Index: 0,
					Delta: types.ChatCompletionStreamChoiceDelta{
						Role:    types.ChatMessageRoleAssistant,
						Content: content,
					},
				}
				
				dataChan <- getStreamResponse(h.ID, choice, h.ModelName)
			}
		}
	}
	
	// 事件结束
	if len(strings.TrimSpace(line)) == 0 && h.IsEvent {
		h.IsEvent = false
		h.CurrentEvent = ""
	}
	
	*rawLine = nil
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
