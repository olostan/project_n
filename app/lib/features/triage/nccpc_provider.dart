/// Project N: NCCPC Pain Observation Checklist Provider.
/// Implements validated psychometric scoring for Non-Communicating Children's Pain Checklist.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/contracts/vocabularies.dart';
import '../auth/auth_provider.dart';

class NCCPCItem {
  final String id;
  final String label;
  final String subscale;

  const NCCPCItem({
    required this.id,
    required this.label,
    required this.subscale,
  });
}

const List<NCCPCItem> canonicalNccpcItems = [
  NCCPCItem(
    id: 'moaning_whining_whimpering',
    label: 'Moaning, whining, whimpering (soft)',
    subscale: 'Vocal',
  ),
  NCCPCItem(id: 'crying', label: 'Crying (moderately loud)', subscale: 'Vocal'),
  NCCPCItem(
    id: 'screaming_yelling',
    label: 'Screaming or yelling (very loud)',
    subscale: 'Vocal',
  ),
  NCCPCItem(
    id: 'less_interaction_withdrawn',
    label: 'Less interaction, withdrawn',
    subscale: 'Social',
  ),
  NCCPCItem(
    id: 'seeking_comfort_closeness',
    label: 'Seeking comfort or physical closeness',
    subscale: 'Social',
  ),
  NCCPCItem(
    id: 'difficult_to_comfort_please',
    label: 'Difficult to comfort or please',
    subscale: 'Social',
  ),
  NCCPCItem(id: 'furrowed_brow', label: 'Furrowed brow', subscale: 'Facial'),
  NCCPCItem(
    id: 'stiff_spastic_rigid_tense',
    label: 'Stiff, spastic, rigid, tense body',
    subscale: 'Body & Limbs',
  ),
  NCCPCItem(
    id: 'protecting_favoring_guarding',
    label: 'Protecting or guarding part of body',
    subscale: 'Body & Limbs',
  ),
  NCCPCItem(
    id: 'flinching_moving_away',
    label: 'Flinching or moving away from touch',
    subscale: 'Body & Limbs',
  ),
];

class NCCPCState {
  final Map<String, int> scores;
  final int totalScore;
  final bool isSubmitting;
  final String? resultMessage;

  const NCCPCState({
    this.scores = const {},
    this.totalScore = 0,
    this.isSubmitting = false,
    this.resultMessage,
  });

  bool get painCutoffBreached => totalScore >= nccpcPvCutoffThreshold;

  NCCPCState copyWith({
    Map<String, int>? scores,
    int? totalScore,
    bool? isSubmitting,
    String? resultMessage,
  }) {
    return NCCPCState(
      scores: scores ?? this.scores,
      totalScore: totalScore ?? this.totalScore,
      isSubmitting: isSubmitting ?? this.isSubmitting,
      resultMessage: resultMessage ?? this.resultMessage,
    );
  }
}

class NCCPCNotifier extends Notifier<NCCPCState> {
  @override
  NCCPCState build() => const NCCPCState();

  void setItemScore(String itemId, int score) {
    final updated = Map<String, int>.from(state.scores);
    updated[itemId] = score;
    final total = updated.values.fold(0, (sum, val) => sum + val);

    state = state.copyWith(scores: updated, totalScore: total);
  }

  void reset() {
    state = const NCCPCState();
  }

  Future<bool> submitTriage(String episodeId) async {
    state = state.copyWith(isSubmitting: true, resultMessage: null);
    try {
      final apiClient = ref.read(apiClientProvider);
      final res = await apiClient.submitNCCPC(
        episodeId: episodeId,
        instrument: 'nccpc_pv',
        scores: state.scores,
      );

      final breached = res['pain_cutoff_breached'] as bool? ?? false;
      final guidance = res['guidance'] as String? ?? 'Checklist recorded.';

      state = state.copyWith(
        isSubmitting: false,
        resultMessage: breached
            ? 'ALERT: NCCPC Pain Cutoff Breached (Score: ${state.totalScore}). $guidance'
            : 'NCCPC Score Recorded (${state.totalScore}/11 threshold).',
      );
      return true;
    } catch (e) {
      state = state.copyWith(
        isSubmitting: false,
        resultMessage: 'Error submitting pain observation checklist.',
      );
      return false;
    }
  }
}

final nccpcProvider = NotifierProvider<NCCPCNotifier, NCCPCState>(
  NCCPCNotifier.new,
);
