import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:project_n_companion/core/storage/outbox_model.dart';
import 'package:project_n_companion/core/storage/outbox_repository.dart';

void main() {
  group('Offline Outbox Tests', () {
    test('QueuedClip serialization and copyWith', () {
      final now = DateTime.now();
      final clip = QueuedClip(
        id: 'clip-123',
        filePath: 'sandbox://clip-123.mp4',
        fileSizeBytes: 2048,
        sha256: 'abc123sha256',
        recordedAt: now,
        antecedentId: 'mealtime',
        totalChunks: 1,
      );

      final map = clip.toMap();
      expect(map['id'], 'clip-123');
      expect(map['status'], 'pending');

      final reconstructed = QueuedClip.fromMap(map);
      expect(reconstructed.id, clip.id);
      expect(reconstructed.status, OutboxStatus.pending);

      final updated = clip.copyWith(status: OutboxStatus.uploaded);
      expect(updated.status, OutboxStatus.uploaded);
    });

    test('InMemoryOutboxRepository lifecycle', () async {
      final repo = InMemoryOutboxRepository();
      final clip = QueuedClip(
        id: 'clip-456',
        filePath: 'sandbox://clip-456.mp4',
        fileSizeBytes: 4096,
        sha256: 'def456sha256',
        recordedAt: DateTime.now(),
        antecedentId: 'loud_environment',
        totalChunks: 2,
      );

      await repo.enqueueClip(clip);
      var pending = await repo.getPendingClips();
      expect(pending.length, 1);
      expect(pending.first.id, 'clip-456');

      await repo.updateClip(clip.copyWith(status: OutboxStatus.uploaded));
      pending = await repo.getPendingClips();
      expect(pending.isEmpty, true);

      await repo.deleteClip('clip-456');
      pending = await repo.getPendingClips();
      expect(pending.isEmpty, true);
    });

    test('SHA-256 integrity calculation', () {
      final bytes = Uint8List.fromList([1, 2, 3, 4, 5]);
      final digest = sha256.convert(bytes).toString();
      expect(digest.length, 64);
      // Deterministic check
      expect(sha256.convert(bytes).toString(), digest);
    });
  });
}
