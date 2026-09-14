/// Project N: Offline Encrypted Outbox Model.
/// Represents a captured observation clip queued for store-and-forward flushing.
library;

enum OutboxStatus { pending, uploading, uploaded, failed }

class QueuedClip {
  final String id;
  final String filePath;
  final int fileSizeBytes;
  final String sha256;
  final DateTime recordedAt;
  final String antecedentId;
  final String? caregiverNotes;
  final OutboxStatus status;
  final int uploadedChunks;
  final int totalChunks;

  const QueuedClip({
    required this.id,
    required this.filePath,
    required this.fileSizeBytes,
    required this.sha256,
    required this.recordedAt,
    required this.antecedentId,
    this.caregiverNotes,
    this.status = OutboxStatus.pending,
    this.uploadedChunks = 0,
    required this.totalChunks,
  });

  QueuedClip copyWith({
    OutboxStatus? status,
    int? uploadedChunks,
    String? caregiverNotes,
  }) {
    return QueuedClip(
      id: id,
      filePath: filePath,
      fileSizeBytes: fileSizeBytes,
      sha256: sha256,
      recordedAt: recordedAt,
      antecedentId: antecedentId,
      caregiverNotes: caregiverNotes ?? this.caregiverNotes,
      status: status ?? this.status,
      uploadedChunks: uploadedChunks ?? this.uploadedChunks,
      totalChunks: totalChunks,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'file_path': filePath,
      'file_size_bytes': fileSizeBytes,
      'sha256': sha256,
      'recorded_at': recordedAt.toIso8601String(),
      'antecedent_id': antecedentId,
      'caregiver_notes': caregiverNotes,
      'status': status.name,
      'uploaded_chunks': uploadedChunks,
      'total_chunks': totalChunks,
    };
  }

  factory QueuedClip.fromMap(Map<String, dynamic> map) {
    return QueuedClip(
      id: map['id'] as String,
      filePath: map['file_path'] as String,
      fileSizeBytes: map['file_size_bytes'] as int,
      sha256: map['sha256'] as String,
      recordedAt: DateTime.parse(map['recorded_at'] as String),
      antecedentId: map['antecedent_id'] as String,
      caregiverNotes: map['caregiver_notes'] as String?,
      status: OutboxStatus.values.byName(map['status'] as String),
      uploadedChunks: map['uploaded_chunks'] as int? ?? 0,
      totalChunks: map['total_chunks'] as int,
    );
  }
}
