/// Project N: Authenticated REST & Streaming API Client.
/// Communicates with local Mac daemon via Bearer authentication over local Wi-Fi.
library;

import 'dart:typed_data';
import 'package:dio/dio.dart';

class ApiClient {
  final Dio dio;
  String? authToken;
  String baseUrl;

  ApiClient({
    Dio? dioClient,
    this.baseUrl = 'http://127.0.0.1:8080',
    this.authToken,
  }) : dio = dioClient ?? Dio() {
    dio.options.baseUrl = baseUrl;
    dio.options.connectTimeout = const Duration(seconds: 10);
    dio.options.receiveTimeout = const Duration(seconds: 15);
  }

  void setToken(String token) {
    authToken = token;
  }

  Options _authOptions() {
    return Options(
      headers: {if (authToken != null) 'Authorization': 'Bearer $authToken'},
    );
  }

  /// Pairs device using local 6-digit PIN.
  Future<String> pairDevice(String deviceId, String pin) async {
    final res = await dio.post(
      '/api/v1/auth/pair',
      data: {
        'device_id': deviceId,
        'csr_pem': '-----BEGIN CERTIFICATE REQUEST-----\n...',
        'pairing_pin': pin,
      },
    );
    final token = res.data['token'] as String;
    setToken(token);
    return token;
  }

  /// Initializes a chunked upload session.
  Future<String> initUpload({
    required String fileName,
    required int fileSize,
    required String sha256,
    required int totalChunks,
  }) async {
    final res = await dio.post(
      '/api/v1/clips/upload/init',
      options: _authOptions(),
      data: {
        'file_name': fileName,
        'file_size': fileSize,
        'sha256': sha256,
        'total_chunks': totalChunks,
      },
    );
    return res.data['upload_id'] as String;
  }

  /// Sends a single binary chunk.
  Future<bool> uploadChunk({
    required String uploadId,
    required int chunkIndex,
    required Uint8List chunkData,
  }) async {
    final res = await dio.put(
      '/api/v1/clips/upload/$uploadId/chunk/$chunkIndex',
      options: _authOptions().copyWith(
        headers: {
          ...?_authOptions().headers,
          'Content-Type': 'application/octet-stream',
        },
      ),
      data: Stream.fromIterable([chunkData]),
    );
    return res.data['received'] as bool? ?? false;
  }

  /// Finalizes upload and encrypts into vault.
  Future<String> finalizeUpload({
    required String uploadId,
    Map<String, dynamic>? metadata,
  }) async {
    final res = await dio.post(
      '/api/v1/clips/upload/$uploadId/finalize',
      options: _authOptions(),
      data: {'metadata_json': metadata ?? {}},
    );
    return res.data['clip_id'] as String;
  }

  /// Triggers background multimodal extraction & neural projection.
  Future<String> triggerAnalysis(String clipId) async {
    final res = await dio.post(
      '/api/v1/clips/$clipId/analyze',
      options: _authOptions(),
      data: {'generate_render': false},
    );
    return res.data['task_id'] as String;
  }

  /// Fetches historical episodes.
  Future<List<Map<String, dynamic>>> fetchEpisodes() async {
    final res = await dio.get('/api/v1/episodes', options: _authOptions());
    final list = res.data['episodes'] as List;
    return list.map((e) => Map<String, dynamic>.from(e as Map)).toList();
  }

  /// Submits caregiver-completed NCCPC pain checklist.
  Future<Map<String, dynamic>> submitNCCPC({
    required String episodeId,
    required String instrument,
    required Map<String, int> scores,
  }) async {
    final res = await dio.post(
      '/api/v1/episodes/$episodeId/nccpc',
      options: _authOptions(),
      data: {'instrument': instrument, 'scores': scores},
    );
    return Map<String, dynamic>.from(res.data as Map);
  }
}
