/// Project N: Episode Timeline Screen.
/// Displays logged episodes, L1-L4 summaries, and co-regulation outcomes.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'timeline_provider.dart';

class TimelineScreen extends ConsumerStatefulWidget {
  const TimelineScreen({super.key});

  @override
  ConsumerState<TimelineScreen> createState() => _TimelineScreenState();
}

class _TimelineScreenState extends ConsumerState<TimelineScreen> {
  @override
  void initState() {
    super.initState();
    Future.microtask(() => ref.read(timelineProvider.notifier).loadEpisodes());
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(timelineProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Observation Timeline'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.read(timelineProvider.notifier).loadEpisodes(),
          ),
        ],
      ),
      body: state.isLoading
          ? const Center(child: CircularProgressIndicator())
          : state.episodes.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.history, size: 64, color: Colors.blueGrey),
                  const SizedBox(height: 16),
                  Text(
                    state.errorMessage ?? 'No episodes logged yet.',
                    style: const TextStyle(color: Colors.black54),
                  ),
                ],
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: state.episodes.length,
              itemBuilder: (context, idx) {
                final ep = state.episodes[idx];
                final antecedent = ep['antecedent_id'] ?? 'unknown';
                final durationSec = ((ep['duration_ms'] ?? 5000) / 1000)
                    .round();
                final action = ep['action_offered'] ?? 'open_observation';
                final outcome = ep['outcome_state'] ?? 'pending review';
                final nccpc = ep['nccpc_score'];

                return Card(
                  margin: const EdgeInsets.only(bottom: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              antecedent
                                  .toString()
                                  .replaceAll('_', ' ')
                                  .toUpperCase(),
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 14,
                                color: Colors.teal,
                              ),
                            ),
                            Text(
                              '${durationSec}s',
                              style: const TextStyle(color: Colors.black54),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Offered: ${action.toString().replaceAll('_', ' ')}',
                        ),
                        Text(
                          'Outcome: ${outcome.toString().replaceAll('_', ' ')}',
                        ),
                        if (nccpc != null) ...[
                          const SizedBox(height: 4),
                          Text(
                            'NCCPC Score: $nccpc / 11',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: (nccpc as int) >= 11
                                  ? Colors.red
                                  : Colors.teal,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }
}
