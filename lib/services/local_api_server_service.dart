import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:get/get.dart';
import 'package:llamadart/llamadart.dart';

import 'chat_storage_service.dart';
import 'llm_service.dart';
import 'wakelock_service.dart';

class LocalApiServerService extends GetxService {
  static const defaultHost = '127.0.0.1';
  static const defaultPort = 4891;

  final LlmService _llm = Get.find<LlmService>();
  final ChatStorageService _storage = Get.find<ChatStorageService>();

  HttpServer? _server;

  final isRunning = false.obs;
  final isStarting = false.obs;
  final errorMessage = ''.obs;
  final port = defaultPort.obs;
  final allInterfaces = false.obs;

  String get host => allInterfaces.value ? '0.0.0.0' : defaultHost;
  String get baseUrl {
    if (allInterfaces.value) {
      return 'http://<device-ip>:${port.value}/v1';
    }
    return 'http://$defaultHost:${port.value}/v1';
  }
  bool get isBusy => _llm.isGenerating.value;
  bool get hasLoadedModel => _llm.isLoaded.value;
  String get modelId => _llm.publicModelId;

  Future<LocalApiServerService> init() async {
    port.value = _normalizePort(_storage.localApiServerPort);
    allInterfaces.value = _storage.localApiAllInterfaces;
    if (_storage.localApiServerEnabled) {
      await start();
    }
    return this;
  }

  Future<void> start({int? requestedPort}) async {
    final nextPort = _normalizePort(requestedPort ?? port.value);
    if (isRunning.value && nextPort == port.value) return;
    if (isRunning.value) {
      await stop(persist: false);
    }

    isStarting.value = true;
    errorMessage.value = '';

    try {
      final bindAddress = allInterfaces.value
          ? InternetAddress.anyIPv4
          : InternetAddress.loopbackIPv4;

      _server = await HttpServer.bind(
        bindAddress,
        nextPort,
        shared: false,
      );
      port.value = nextPort;
      _storage.localApiServerPort = nextPort;
      _storage.localApiServerEnabled = true;
      isRunning.value = true;

      try {
        final wakelockService = Get.find<WakelockService>();
        await wakelockService.enableForInference();
      } catch (_) {}

      unawaited(
        _server!
            .listen(
              _handleRequest,
              onError: (Object error) {
                errorMessage.value = error.toString();
              },
            )
            .asFuture<void>(),
      );
    } catch (e) {
      _server = null;
      isRunning.value = false;
      errorMessage.value = e.toString();
      // Do not rethrow here, so that app initialization can continue 
      // even if the local API server fails to bind.
    } finally {
      isStarting.value = false;
    }
  }

  Future<void> stop({bool persist = true}) async {
    final server = _server;
    _server = null;
    if (server != null) {
      await server.close(force: true);
    }
    isRunning.value = false;
    if (persist) {
      _storage.localApiServerEnabled = false;
    }
    try {
      final wakelockService = Get.find<WakelockService>();
      await wakelockService.disable();
    } catch (_) {}
  }

  Future<void> setPort(int nextPort) async {
    final normalized = _normalizePort(nextPort);
    port.value = normalized;
    _storage.localApiServerPort = normalized;
    if (isRunning.value) {
      await start(requestedPort: normalized);
    }
  }

  Future<void> setAllInterfaces(bool value) async {
    allInterfaces.value = value;
    _storage.localApiAllInterfaces = value;
    if (isRunning.value) {
      // Restart with new binding
      await start();
    }
  }

  /// Get the device's local network IP address.
  Future<String?> getDeviceIp() async {
    try {
      final interfaces = await NetworkInterface.list(
        type: InternetAddressType.IPv4,
        includeLoopback: false,
      );
      for (final iface in interfaces) {
        for (final addr in iface.addresses) {
          if (!addr.isLoopback) {
            return addr.address;
          }
        }
      }
    } catch (_) {}
    return null;
  }

  int _normalizePort(int value) {
    if (value < 1024 || value > 65535) return defaultPort;
    return value;
  }

  Future<void> _handleRequest(HttpRequest request) async {
    _applyCorsHeaders(request.response);

    if (request.method == 'OPTIONS') {
      request.response.statusCode = HttpStatus.noContent;
      await request.response.close();
      return;
    }

    try {
      final path = request.uri.path;
      if (request.method == 'GET' && path == '/healthz') {
        await _writeJson(request.response, _healthJson());
        return;
      }

      if (request.method == 'GET' && path == '/v1/models') {
        await _writeJson(request.response, _modelsJson());
        return;
      }

      if (request.method == 'POST' && path == '/v1/chat/completions') {
        await _handleChatCompletions(request);
        return;
      }

      await _writeError(
        request.response,
        HttpStatus.notFound,
        'No route for `${request.method} $path`.',
        type: 'invalid_request_error',
        code: 'not_found',
      );
    } on _OpenAiRequestException catch (e) {
      await _writeError(
        request.response,
        HttpStatus.badRequest,
        e.message,
        type: 'invalid_request_error',
        param: e.param,
      );
    } catch (e) {
      await _writeError(
        request.response,
        HttpStatus.internalServerError,
        'Unexpected server error: $e',
        type: 'server_error',
      );
    }
  }

  Map<String, dynamic> _healthJson() {
    return {
      'status': 'ok',
      'ready': hasLoadedModel,
      'model': hasLoadedModel ? modelId : null,
      'busy': isBusy,
      'host': host,
      'port': port.value,
      'base_url': baseUrl,
    };
  }

  Map<String, dynamic> _modelsJson() {
    final data = hasLoadedModel
        ? [
            {
              'id': modelId,
              'object': 'model',
              'created': 0,
              'owned_by': 'uncensored-local-ai',
            },
          ]
        : <Map<String, dynamic>>[];

    return {'object': 'list', 'data': data};
  }

  Future<void> _handleChatCompletions(HttpRequest request) async {
    if (!hasLoadedModel) {
      await _writeError(
        request.response,
        HttpStatus.serviceUnavailable,
        'No model loaded. Load a model in Uncensored Local AI first.',
        type: 'invalid_request_error',
        code: 'model_not_loaded',
      );
      return;
    }

    if (isBusy) {
      await _writeError(
        request.response,
        HttpStatus.tooManyRequests,
        'Another generation is already in progress. Retry shortly.',
        type: 'server_error',
        code: 'busy',
      );
      return;
    }

    final body = await _readJsonObject(request);
    final chatRequest = _parseChatCompletionRequest(body);

    if (chatRequest.stream) {
      await _streamChatCompletion(request.response, chatRequest);
      return;
    }

    await _writeJson(
      request.response,
      await _createChatCompletion(chatRequest),
    );
  }

  Future<Map<String, dynamic>> _createChatCompletion(
    _ChatCompletionRequest request,
  ) async {
    final created = _unixSeconds();
    final id = _completionId();
    final buffer = StringBuffer();

    await for (final token in _llm.generateChatCompletion(
      messages: request.messages,
      params: request.params,
    )) {
      buffer.write(token);
    }

    final content = buffer.toString().trim();
    final usage = await _usageJson(request.messages, content);

    return {
      'id': id,
      'object': 'chat.completion',
      'created': created,
      'model': modelId,
      'choices': [
        {
          'index': 0,
          'message': {'role': 'assistant', 'content': content},
          'finish_reason': 'stop',
        },
      ],
      'usage': usage,
    };
  }

  Future<void> _streamChatCompletion(
    HttpResponse response,
    _ChatCompletionRequest request,
  ) async {
    final created = _unixSeconds();
    final id = _completionId();

    response.statusCode = HttpStatus.ok;
    response.headers
      ..contentType = ContentType('text', 'event-stream', charset: 'utf-8')
      ..set(HttpHeaders.cacheControlHeader, 'no-cache')
      ..set(HttpHeaders.connectionHeader, 'keep-alive');

    void writeEvent(Map<String, dynamic> payload) {
      response.write('data: ${jsonEncode(payload)}\n\n');
    }

    writeEvent(
      _streamChunk(id: id, created: created, delta: {'role': 'assistant'}),
    );

    try {
      await for (final token in _llm.generateChatCompletion(
        messages: request.messages,
        params: request.params,
      )) {
        if (token.isEmpty) continue;
        writeEvent(
          _streamChunk(id: id, created: created, delta: {'content': token}),
        );
      }

      writeEvent(
        _streamChunk(
          id: id,
          created: created,
          delta: <String, dynamic>{},
          finishReason: 'stop',
        ),
      );
      response.write('data: [DONE]\n\n');
    } catch (e) {
      writeEvent({
        'error': {
          'message': 'Model generation failed: $e',
          'type': 'server_error',
          'param': null,
          'code': 'generation_failed',
        },
      });
      response.write('data: [DONE]\n\n');
    } finally {
      await response.close();
    }
  }

  Map<String, dynamic> _streamChunk({
    required String id,
    required int created,
    required Map<String, dynamic> delta,
    String? finishReason,
  }) {
    return {
      'id': id,
      'object': 'chat.completion.chunk',
      'created': created,
      'model': modelId,
      'choices': [
        {'index': 0, 'delta': delta, 'finish_reason': finishReason},
      ],
    };
  }

  _ChatCompletionRequest _parseChatCompletionRequest(
    Map<String, dynamic> body,
  ) {
    if (body['tools'] is List && (body['tools'] as List).isNotEmpty) {
      throw _OpenAiRequestException(
        'Tool calling is not supported by this local API server yet.',
        param: 'tools',
      );
    }

    final rawMessages = body['messages'];
    if (rawMessages is! List || rawMessages.isEmpty) {
      throw _OpenAiRequestException(
        '`messages` must be a non-empty array.',
        param: 'messages',
      );
    }

    final messages = rawMessages
        .map((raw) => _parseMessage(raw))
        .toList(growable: false);

    final maxTokens =
        _readInt(body['max_completion_tokens'], 'max_completion_tokens') ??
        _readInt(body['max_tokens'], 'max_tokens');

    var params = const GenerationParams(penalty: 1.0, topP: 0.95, minP: 0.05);

    if (maxTokens != null) {
      params = params.copyWith(maxTokens: maxTokens);
    }

    final temperature = _readDouble(body['temperature'], 'temperature');
    if (temperature != null) {
      params = params.copyWith(temp: temperature);
    }

    final topP = _readDouble(body['top_p'], 'top_p');
    if (topP != null) {
      params = params.copyWith(topP: topP);
    }

    final seed = _readInt(body['seed'], 'seed');
    if (seed != null) {
      params = params.copyWith(seed: seed);
    }

    final stops = _parseStop(body['stop']);
    if (stops.isNotEmpty) {
      params = params.copyWith(stopSequences: stops);
    }

    return _ChatCompletionRequest(
      messages: messages,
      params: params,
      stream: _readBool(body['stream'], 'stream') ?? false,
    );
  }

  LlamaChatMessage _parseMessage(Object? raw) {
    if (raw is! Map) {
      throw _OpenAiRequestException('Each message must be an object.');
    }

    final roleRaw = raw['role'];
    final contentRaw = raw['content'];
    if (roleRaw is! String || roleRaw.trim().isEmpty) {
      throw _OpenAiRequestException(
        'Message role must be a non-empty string.',
        param: 'messages.role',
      );
    }

    final role = switch (roleRaw) {
      'developer' || 'system' => LlamaChatRole.system,
      'user' => LlamaChatRole.user,
      'assistant' => LlamaChatRole.assistant,
      'tool' => LlamaChatRole.tool,
      _ => throw _OpenAiRequestException(
        'Unsupported message role `$roleRaw`.',
        param: 'messages.role',
      ),
    };

    final content = _parseMessageContent(contentRaw);
    return LlamaChatMessage.fromText(role: role, text: content);
  }

  String _parseMessageContent(Object? raw) {
    if (raw == null) return '';
    if (raw is String) return raw;
    if (raw is List) {
      final buffer = StringBuffer();
      for (final part in raw) {
        if (part is! Map) {
          throw _OpenAiRequestException(
            'Message content parts must be objects.',
          );
        }
        final type = part['type'];
        if (type == 'text' && part['text'] is String) {
          buffer.write(part['text'] as String);
        } else {
          throw _OpenAiRequestException(
            'Only text message content is supported by this local API server.',
            param: 'messages.content',
          );
        }
      }
      return buffer.toString();
    }
    throw _OpenAiRequestException(
      'Message content must be a string or text content parts.',
      param: 'messages.content',
    );
  }

  List<String> _parseStop(Object? raw) {
    if (raw == null) return const [];
    if (raw is String) return [raw];
    if (raw is List && raw.every((item) => item is String)) {
      return raw.cast<String>();
    }
    throw _OpenAiRequestException(
      '`stop` must be a string or an array of strings.',
      param: 'stop',
    );
  }

  int? _readInt(Object? raw, String param) {
    if (raw == null) return null;
    if (raw is int) return raw;
    throw _OpenAiRequestException('`$param` must be an integer.', param: param);
  }

  double? _readDouble(Object? raw, String param) {
    if (raw == null) return null;
    if (raw is num) return raw.toDouble();
    throw _OpenAiRequestException('`$param` must be a number.', param: param);
  }

  bool? _readBool(Object? raw, String param) {
    if (raw == null) return null;
    if (raw is bool) return raw;
    throw _OpenAiRequestException('`$param` must be a boolean.', param: param);
  }

  Future<Map<String, dynamic>> _readJsonObject(HttpRequest request) async {
    try {
      final raw = await utf8.decoder.bind(request).join();
      final decoded = jsonDecode(raw);
      if (decoded is Map<String, dynamic>) return decoded;
      if (decoded is Map) return Map<String, dynamic>.from(decoded);
    } catch (_) {
      throw _OpenAiRequestException('Request body must be valid JSON.');
    }
    throw _OpenAiRequestException('Request body must be a JSON object.');
  }

  Future<Map<String, int>> _usageJson(
    List<LlamaChatMessage> messages,
    String completion,
  ) async {
    final promptText = messages
        .map((m) => '${m.role.name}: ${m.content}')
        .join('\n');
    final promptTokens = await _llm.countTokens(promptText);
    final completionTokens = await _llm.countTokens(completion);
    return {
      'prompt_tokens': promptTokens,
      'completion_tokens': completionTokens,
      'total_tokens': promptTokens + completionTokens,
    };
  }

  Future<void> _writeJson(
    HttpResponse response,
    Map<String, dynamic> body, {
    int statusCode = HttpStatus.ok,
  }) async {
    response.headers.contentType ??= ContentType.json;
    response.statusCode = statusCode;
    response.write(jsonEncode(body));
    await response.close();
  }

  Future<void> _writeError(
    HttpResponse response,
    int statusCode,
    String message, {
    String type = 'invalid_request_error',
    String? code,
    String? param,
  }) {
    return _writeJson(response, {
      'error': {'message': message, 'type': type, 'param': param, 'code': code},
    }, statusCode: statusCode);
  }

  void _applyCorsHeaders(HttpResponse response) {
    response.headers
      ..set(HttpHeaders.accessControlAllowOriginHeader, '*')
      ..set(HttpHeaders.accessControlAllowMethodsHeader, 'GET, POST, OPTIONS')
      ..set(
        HttpHeaders.accessControlAllowHeadersHeader,
        'authorization, content-type',
      );
  }

  int _unixSeconds() => DateTime.now().millisecondsSinceEpoch ~/ 1000;

  String _completionId() => 'chatcmpl-${DateTime.now().microsecondsSinceEpoch}';

  @override
  Future<void> onClose() async {
    await stop();
    super.onClose();
  }
}

class _ChatCompletionRequest {
  final List<LlamaChatMessage> messages;
  final GenerationParams params;
  final bool stream;

  const _ChatCompletionRequest({
    required this.messages,
    required this.params,
    required this.stream,
  });
}

class _OpenAiRequestException implements Exception {
  final String message;
  final String? param;

  const _OpenAiRequestException(this.message, {this.param});
}
