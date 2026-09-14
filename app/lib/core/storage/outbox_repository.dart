/// Project N: Offline Outbox Repository.
/// Local store-and-forward queue manager for pending observation clips.
library;

import 'outbox_model.dart';

abstract class OutboxRepository {
  Future<void> enqueueClip(QueuedClip clip);
  Future<List<QueuedClip>> getPendingClips();
  Future<void> updateClip(QueuedClip clip);
  Future<void> deleteClip(String id);
}

class InMemoryOutboxRepository implements OutboxRepository {
  final Map<String, QueuedClip> _queue = {};

  @override
  Future<void> enqueueClip(QueuedClip clip) async {
    _queue[clip.id] = clip;
  }

  @override
  Future<List<QueuedClip>> getPendingClips() async {
    return _queue.values
        .where((c) => c.status != OutboxStatus.uploaded)
        .toList();
  }

  @override
  Future<void> updateClip(QueuedClip clip) async {
    _queue[clip.id] = clip;
  }

  @override
  Future<void> deleteClip(String id) async {
    _queue.remove(id);
  }
}
