package replicate

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"one-api/common"
	"one-api/types"
	"strings"
)

// 转换OpenAI请求到Replicate请求格式
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
				// 处理图片URL
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

// 从Replicate响应转换为OpenAI格式
func (p *ReplicateProvider) convertToChatOpenai(response *ReplicateResponse[[]string]) (*types.ChatCompletionResponse, *types.OpenAIErrorWithStatusCode) {
	// 直接拼接所有文本片段，保留原始格式
	var responseText string
	if response.Output != nil {
		responseText = strings.Join(response.Output, "")
	}

	// 创建完成响应
	choice := types.ChatCompletionChoice{
		Message: types.ChatCompletionMessage{
			Role:    types.ChatMessageRoleAssistant,
			Content: responseText,
		},
		FinishReason: types.FinishReasonStop,
	}

	resp := &types.ChatCompletionResponse{
		ID:      response.ID,
		Object:  "chat.completion",
		Created: common.GetTimestamp(),
		Model:   p.ModelName,
		Choices: []types.ChatCompletionChoice{choice},
	}

	// 添加使用统计
	if response.Metrics != nil {
		resp.Usage = types.Usage{
			PromptTokens:     response.Metrics.InputTokenCount,
			CompletionTokens: response.Metrics.OutputTokenCount,
			TotalTokens:      response.Metrics.InputTokenCount + response.Metrics.OutputTokenCount,
		}
	}

	return resp, nil
}

// 执行聊天完成请求
func (p *ReplicateProvider) ChatCompletion(request *types.ChatCompletionRequest, apiKey string) (*types.ChatCompletionResponse, *types.OpenAIErrorWithStatusCode) {
	// 转换请求格式
	replicateRequest := convertFromChatOpenai(request)

	// 准备API请求
	jsonData, err := json.Marshal(replicateRequest)
	if err != nil {
		return nil, common.ErrorWrapper(err, "json_marshal_error", http.StatusInternalServerError)
	}

	// 构建请求URL
	requestURL := p.HostName + "/v1/predictions"

	// 创建HTTP请求
	req, err := http.NewRequest("POST", requestURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, common.ErrorWrapper(err, "create_request_error", http.StatusInternalServerError)
	}

	// 设置请求头
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Token %s", apiKey))

	// 发送请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, common.ErrorWrapper(err, "send_request_error", http.StatusInternalServerError)
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusAccepted {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, &types.OpenAIErrorWithStatusCode{
			OpenAIError: types.OpenAIError{
				Message: fmt.Sprintf("API request error, status: %d, message: %s", resp.StatusCode, string(bodyBytes)),
				Type:    "replicate_error",
			},
			StatusCode: resp.StatusCode,
		}
	}

	// 解析响应
	if request.Stream {
		// 对于流响应，我们需要采用不同的处理方法
		// 此处仅返回初始响应，实际流处理在 ChatCompletionStream 中
		var replicateResp ReplicateResponse[interface{}]
		if err := json.NewDecoder(resp.Body).Decode(&replicateResp); err != nil {
			return nil, common.ErrorWrapper(err, "decode_error", http.StatusInternalServerError)
		}
		
		// 转换为 OpenAI 流式响应格式
		streamURL, ok := replicateResp.URLs.Stream.(string)
		if !ok || streamURL == "" {
			return nil, common.ErrorWrapper(fmt.Errorf("missing stream URL"), "missing_stream_url", http.StatusInternalServerError)
		}
		
		// 创建一个基本响应，客户端将使用流式处理
		openaiResp := &types.ChatCompletionResponse{
			ID:      replicateResp.ID,
			Object:  "chat.completion",
			Created: common.GetTimestamp(),
			Model:   p.ModelName,
			Choices: []types.ChatCompletionChoice{{
				Message: types.ChatCompletionMessage{
					Role:    types.ChatMessageRoleAssistant,
					Content: "", // 内容将在流中提供
				},
				FinishReason: types.FinishReasonNull,
			}},
		}
		
		return openaiResp, nil
	}

	// 非流式处理
	var replicateResp ReplicateResponse[[]string]
	if err := json.NewDecoder(resp.Body).Decode(&replicateResp); err != nil {
		return nil, common.ErrorWrapper(err, "decode_error", http.StatusInternalServerError)
	}

	// 检查请求是否成功
	if replicateResp.Status != "succeeded" {
		// 如果状态不是成功，尝试轮询结果
		replicateResp, err = p.pollResult(replicateResp.ID, apiKey)
		if err != nil {
			return nil, common.ErrorWrapper(err, "polling_error", http.StatusInternalServerError)
		}
	}

	// 转换成功的响应
	return p.convertToChatOpenai(&replicateResp)
}

// 流式聊天处理
func (p *ReplicateProvider) ChatCompletionStream(request *types.ChatCompletionRequest, apiKey string, writer io.Writer) *types.OpenAIErrorWithStatusCode {
	chatResp, err := p.ChatCompletion(request, apiKey)
	if err != nil {
		return err
	}

	// 获取流URL
	predictionId := chatResp.ID
	streamUrl, streamErr := p.getStreamUrl(predictionId, apiKey)
	if streamErr != nil {
		return streamErr
	}

	// 处理流
	return p.handleChatCompletionStream(streamUrl, writer)
}

// 处理聊天完成流
func (p *ReplicateProvider) handleChatCompletionStream(streamUrl string, writer io.Writer) *types.OpenAIErrorWithStatusCode {
	// 创建HTTP请求
	req, err := http.NewRequest("GET", streamUrl, nil)
	if err != nil {
		return common.ErrorWrapper(err, "create_stream_request_error", http.StatusInternalServerError)
	}

	// 发送请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return common.ErrorWrapper(err, "stream_request_error", http.StatusInternalServerError)
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return &types.OpenAIErrorWithStatusCode{
			OpenAIError: types.OpenAIError{
				Message: fmt.Sprintf("Stream request error, status: %d, message: %s", resp.StatusCode, string(bodyBytes)),
				Type:    "replicate_stream_error",
			},
			StatusCode: resp.StatusCode,
		}
	}

	// 设置流处理器
	handler := NewReplicateStreamHandler(writer)
	reader := bufio.NewReader(resp.Body)

	// 读取SSE流
	var buffer bytes.Buffer
	dataChan := make(chan string)
	errChan := make(chan error)

	// 启动一个goroutine来处理收到的数据
	go func() {
		for {
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				errChan <- err
				return
			}

			// 忽略空行
			if len(line) <= 2 {
				continue
			}

			// 检查是否是data前缀
			if bytes.HasPrefix(line, []byte("data: ")) {
				handler.HandlerChatStream(&line, dataChan, errChan)
			}
		}
		close(dataChan)
	}()

	// 处理从goroutine接收的数据
	for {
		select {
		case data, ok := <-dataChan:
			if !ok {
				// 通道已关闭，流处理完成
				return nil
			}
			buffer.WriteString(data)
		case err := <-errChan:
			return common.ErrorWrapper(err, "stream_processing_error", http.StatusInternalServerError)
		}
	}
}

// 获取流式URL
func (p *ReplicateProvider) getStreamUrl(predictionId string, apiKey string) (string, *types.OpenAIErrorWithStatusCode) {
	// 构建请求URL
	requestURL := p.HostName + "/v1/predictions/" + predictionId

	// 创建HTTP请求
	req, err := http.NewRequest("GET", requestURL, nil)
	if err != nil {
		return "", common.ErrorWrapper(err, "create_request_error", http.StatusInternalServerError)
	}

	// 设置请求头
	req.Header.Set("Authorization", fmt.Sprintf("Token %s", apiKey))

	// 发送请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", common.ErrorWrapper(err, "send_request_error", http.StatusInternalServerError)
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", &types.OpenAIErrorWithStatusCode{
			OpenAIError: types.OpenAIError{
				Message: fmt.Sprintf("API request error, status: %d, message: %s", resp.StatusCode, string(bodyBytes)),
				Type:    "replicate_error",
			},
			StatusCode: resp.StatusCode,
		}
	}

	// 解析响应
	var replicateResp ReplicateResponse[interface{}]
	if err := json.NewDecoder(resp.Body).Decode(&replicateResp); err != nil {
		return "", common.ErrorWrapper(err, "decode_error", http.StatusInternalServerError)
	}

	// 获取流URL
	if replicateResp.URLs.Stream == "" {
		return "", common.ErrorWrapper(fmt.Errorf("missing stream URL"), "missing_stream_url", http.StatusInternalServerError)
	}

	return replicateResp.URLs.Stream, nil
}

// 轮询获取结果
func (p *ReplicateProvider) pollResult(predictionId string, apiKey string) (ReplicateResponse[[]string], error) {
	var result ReplicateResponse[[]string]
	
	// 构建请求URL
	requestURL := p.HostName + "/v1/predictions/" + predictionId

	// 设置最大轮询次数
	maxAttempts := 30
	for i := 0; i < maxAttempts; i++ {
		// 创建HTTP请求
		req, err := http.NewRequest("GET", requestURL, nil)
		if err != nil {
			return result, err
		}

		// 设置请求头
		req.Header.Set("Authorization", fmt.Sprintf("Token %s", apiKey))

		// 发送请求
		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			return result, err
		}

		// 解析响应
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			resp.Body.Close()
			return result, err
		}
		resp.Body.Close()

		// 检查状态
		if result.Status == "succeeded" {
			return result, nil
		} else if result.Status == "failed" || result.Status == "canceled" {
			return result, fmt.Errorf("prediction failed or canceled: %s", result.Error)
		}

		// 等待一段时间后重试
		time.Sleep(1 * time.Second)
	}

	return result, fmt.Errorf("polling timeout after %d attempts", maxAttempts)
}
