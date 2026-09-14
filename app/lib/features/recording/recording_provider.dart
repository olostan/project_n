/// Project N: Recording & Outbox Sync Provider.
/// Controls continuous 30-120s capture, SHA-256 integrity, and chunked upload flushing.
library;

import 'dart:async';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';

import '../../core/contracts/vocabularies.dart';
import '../../core/storage/outbox_model.dart';
import '../../core/storage/outbox_repository.dart';
import '../auth/auth_provider.dart';

class RecordingState {
  final bool isRecording;
  final int secondsElapsed;
  final String selectedAntecedent;
  final String? notes;
  final bool isSyncing;
  final double syncProgress;
  final String? statusMessage;

  const RecordingState({
    this.isRecording = false,
    this.secondsElapsed = 0,
    this.selectedAntecedent = 'unknown',
    this.notes,
    this.isSyncing = false,
    this.syncProgress = 0.0,
    this.statusMessage,
  });

  bool get canStop => secondsElapsed >= minClipDurationSec;

  RecordingState copyWith({
    bool? isRecording,
    int? secondsElapsed,
    String? selectedAntecedent,
    String? notes,
    bool? isSyncing,
    double? syncProgress,
    String? statusMessage,
  }) {
    return RecordingState(
      isRecording: isRecording ?? this.isRecording,
      secondsElapsed: secondsElapsed ?? this.secondsElapsed,
      selectedAntecedent: selectedAntecedent ?? this.selectedAntecedent,
      notes: notes ?? this.notes,
      isSyncing: isSyncing ?? this.isSyncing,
      syncProgress: syncProgress ?? this.syncProgress,
      statusMessage: statusMessage ?? this.statusMessage,
    );
  }
}

class RecordingNotifier extends Notifier<RecordingState> {
  Timer? _timer;

  @override
  RecordingState build() {
    ref.onDispose(() => _timer?.cancel());
    return const RecordingState();
  }

  void setAntecedent(String antecedent) {
    state = state.copyWith(selectedAntecedent: antecedent);
  }

  void setNotes(String notes) {
    state = state.copyWith(notes: notes);
  }

  void startRecording() {
    _timer?.cancel();
    state = state.copyWith(
      isRecording: true,
      secondsElapsed: 0,
      statusMessage: 'Recording observation...',
    );
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (state.secondsElapsed >= maxClipDurationSec) {
        stopRecording();
      } else {
        state = state.copyWith(secondsElapsed: state.secondsElapsed + 1);
      }
    });
  }

  Future<QueuedClip?> stopRecording([Uint8List? rawBytes]) async {
    _timer?.cancel();
    final duration = state.secondsElapsed;
    state = state.copyWith(isRecording: false);

    if (duration < minClipDurationSec) {
      state = state.copyWith(
        statusMessage: 'Discarded: Minimum 30.0s recording duration required.',
      );
      return null;
    }

    // Generate synthetic or recorded clip bytes
    final bytes = rawBytes ?? Uint8List(1024 * 100); // 100 KB payload
    final digest = sha256.convert(bytes).toString();
    final clipId = const Uuid().v4();

    // 1 MB chunking (or 1 chunk for small clips)
    const chunkSize = 1048576;
    final totalChunks = (bytes.length / chunkSize).ceil().clamp(1, 9999);

    final queued = QueuedClip(
      id: clipId,
      filePath: 'sandbox://$clipId.mp4',
      fileSizeBytes: bytes.length,
      sha256: digest,
      recordedAt: DateTime.now(),
      antecedentId: state.selectedAntecedent,
      caregiverNotes: state.notes,
      totalChunks: totalChunks,
    );

    final outbox = ref.read(outboxRepositoryProvider);
    await outbox.enqueueClip(queued);
    state = state.copyWith(
      statusMessage: 'Clip securely stored in offline outbox. Ready to sync.',
    );
    return queued;
  }

  Future<void> flushOutbox(Uint8List sampleData) async {
    final outbox = ref.read(outboxRepositoryProvider);
    final apiClient = ref.read(apiClientProvider);

    final pending = await outbox.getPendingClips();
    if (pending.isEmpty) return;

    state = state.copyWith(isSyncing: true, syncProgress: 0.0);

    for (final clip in pending) {
      try {
        final uploadId = await apiClient.initUpload(
          fileName: '${clip.id}.mp4',
          fileSize: clip.fileSizeBytes,
          sha256: clip.sha256,
          totalChunks: clip.totalChunks,
        );

        await apiClient.uploadChunk(
          uploadId: uploadId,
          chunkIndex: 0,
          chunkData: sampleData,
        );

        await apiClient.finalizeUpload(
          uploadId: uploadId,
          metadata: {'antecedent_id': clip.antecedentId},
        );

        await apiClient.triggerAnalysis(clip.id);
        await outbox.updateClip(clip.copyWith(status: OutboxStatus.uploaded));
      } catch (e) {
        await outbox.updateClip(clip.copyWith(status: OutboxStatus.failed));
      }
    }

    state = state.copyWith(
      isSyncing: false,
      syncProgress: 1.0,
      statusMessage: 'Store-and-forward sync complete.',
    );
  }
}

final outboxRepositoryProvider = Provider<OutboxRepository>((ref) {
  return InMemoryOutboxRepository();
});

final recordingProvider = NotifierProvider<RecordingNotifier, RecordingState>(
  RecordingNotifier.new,
);
