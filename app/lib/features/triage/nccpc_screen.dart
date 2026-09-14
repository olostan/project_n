/// Project N: NCCPC Pain Checklist Screen.
/// Validated 10-minute observational pain assessment with threshold alerting.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'nccpc_provider.dart';

class NCCPCScreen extends ConsumerWidget {
  final String episodeId;

  const NCCPCScreen({super.key, this.episodeId = 'latest_episode'});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final nccpcState = ref.watch(nccpcProvider);
    final nccpcNotifier = ref.read(nccpcProvider.notifier);

    return Scaffold(
      appBar: AppBar(
        title: const Text('NCCPC-PV Pain Checklist'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => nccpcNotifier.reset(),
          ),
        ],
      ),
      body: Column(
        children: [
          // Total Score Banner
          Container(
            padding: const EdgeInsets.all(16),
            color: nccpcState.painCutoffBreached
                ? Colors.red.shade100
                : Colors.teal.shade50,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Total Score: ${nccpcState.totalScore} / 11 Threshold',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: nccpcState.painCutoffBreached
                            ? Colors.red.shade900
                            : Colors.teal.shade900,
                      ),
                    ),
                    Text(
                      nccpcState.painCutoffBreached
                          ? 'PAIN CUTOFF BREACHED: Medical triage precedence'
                          : 'Score below significant pain cutoff',
                      style: TextStyle(
                        fontSize: 12,
                        color: nccpcState.painCutoffBreached
                            ? Colors.red.shade800
                            : Colors.teal.shade800,
                      ),
                    ),
                  ],
                ),
                ElevatedButton(
                  onPressed: nccpcState.isSubmitting
                      ? null
                      : () => nccpcNotifier.submitTriage(episodeId),
                  child: const Text('Submit'),
                ),
              ],
            ),
          ),

          if (nccpcState.resultMessage != null)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                nccpcState.resultMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ),

          // Items List
          Expanded(
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: canonicalNccpcItems.length,
              separatorBuilder: (_, _) => const Divider(),
              itemBuilder: (context, idx) {
                final item = canonicalNccpcItems[idx];
                final currentVal = nccpcState.scores[item.id] ?? 0;

                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '[${item.subscale}] ${item.label}',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _ScoreButton(
                          label: '0 (None)',
                          score: 0,
                          selected: currentVal == 0,
                          onTap: () => nccpcNotifier.setItemScore(item.id, 0),
                        ),
                        _ScoreButton(
                          label: '1 (Little)',
                          score: 1,
                          selected: currentVal == 1,
                          onTap: () => nccpcNotifier.setItemScore(item.id, 1),
                        ),
                        _ScoreButton(
                          label: '2 (Often)',
                          score: 2,
                          selected: currentVal == 2,
                          onTap: () => nccpcNotifier.setItemScore(item.id, 2),
                        ),
                        _ScoreButton(
                          label: '3 (Very)',
                          score: 3,
                          selected: currentVal == 3,
                          onTap: () => nccpcNotifier.setItemScore(item.id, 3),
                        ),
                      ],
                    ),
                  ],
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _ScoreButton extends StatelessWidget {
  final String label;
  final int score;
  final bool selected;
  final VoidCallback onTap;

  const _ScoreButton({
    required this.label,
    required this.score,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return ChoiceChip(
      label: Text(label),
      selected: selected,
      onSelected: (_) => onTap(),
    );
  }
}
