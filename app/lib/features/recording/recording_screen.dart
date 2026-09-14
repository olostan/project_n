/// Project N: Sensory Recording & Outbox Screen.
/// Captures 30-120s observations and provides offline-outbox sync status.
library;

import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/contracts/vocabularies.dart';
import 'recording_provider.dart';

class RecordingScreen extends ConsumerWidget {
  const RecordingScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final recState = ref.watch(recordingProvider);
    final recNotifier = ref.read(recordingProvider.notifier);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Capture Observation'),
        actions: [
          IconButton(
            icon: const Icon(Icons.sync),
            tooltip: 'Flush Outbox',
            onPressed: recState.isSyncing
                ? null
                : () {
                    // Send sample synthetic data for demo/testing
                    recNotifier.flushOutbox(Uint8List(1024 * 50));
                  },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Timer & Status Display
            Card(
              elevation: 2,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  children: [
                    Icon(
                      recState.isRecording
                          ? Icons.fiber_manual_record
                          : Icons.videocam_outlined,
                      color: recState.isRecording
                          ? Colors.red
                          : Colors.blueGrey,
                      size: 48,
                    ),
                    const SizedBox(height: 12),
                    Text(
                      '${recState.secondsElapsed}s / 120s',
                      style: const TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      recState.isRecording
                          ? (recState.canStop
                                ? 'Minimum duration reached. Ready to stop.'
                                : 'Recording: Need at least 30.0s.')
                          : 'Press Start to begin a 30–120s observation clip.',
                      style: TextStyle(
                        color: recState.isRecording && !recState.canStop
                            ? Colors.orange.shade800
                            : Colors.black54,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),

            // Antecedent Selector
            DropdownButtonFormField<String>(
              initialValue: recState.selectedAntecedent,
              decoration: const InputDecoration(
                labelText: 'Situational Context / Antecedent',
                border: OutlineInputBorder(),
              ),
              items: controlledAntecedents.map((ant) {
                return DropdownMenuItem(
                  value: ant,
                  child: Text(ant.replaceAll('_', ' ')),
                );
              }).toList(),
              onChanged: recState.isRecording
                  ? null
                  : (val) {
                      if (val != null) recNotifier.setAntecedent(val);
                    },
            ),
            const SizedBox(height: 16),

            // Quick Caregiver Notes
            TextField(
              decoration: const InputDecoration(
                labelText: 'Optional Caregiver Notes',
                hintText: 'e.g., loud siren outside, refused coat...',
                border: OutlineInputBorder(),
              ),
              maxLines: 2,
              onChanged: (text) => recNotifier.setNotes(text),
            ),
            const SizedBox(height: 24),

            // Record / Stop Action Button
            if (!recState.isRecording)
              ElevatedButton.icon(
                icon: const Icon(Icons.play_arrow),
                label: const Text('Start Recording (30–120s)'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.teal.shade700,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                onPressed: () => recNotifier.startRecording(),
              )
            else
              ElevatedButton.icon(
                icon: const Icon(Icons.stop),
                label: const Text('Stop & Queue in Outbox'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: recState.canStop
                      ? Colors.red.shade700
                      : Colors.grey,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                onPressed: () => recNotifier.stopRecording(),
              ),

            const SizedBox(height: 20),
            if (recState.statusMessage != null)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.blueGrey.shade50,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  recState.statusMessage!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontWeight: FontWeight.w500),
                ),
              ),
            if (recState.isSyncing) ...[
              const SizedBox(height: 12),
              const LinearProgressIndicator(),
            ],
          ],
        ),
      ),
    );
  }
}
