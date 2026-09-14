/// Project N: Episode Timeline Provider.
/// Manages verified episode history, 4-layer outputs, and outcome tracking.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../auth/auth_provider.dart';

class TimelineState {
  final bool isLoading;
  final List<Map<String, dynamic>> episodes;
  final String? errorMessage;
  final String? selectedEpisodeId;

  const TimelineState({
    this.isLoading = false,
    this.episodes = const [],
    this.errorMessage,
    this.selectedEpisodeId,
  });

  TimelineState copyWith({
    bool? isLoading,
    List<Map<String, dynamic>>? episodes,
    String? errorMessage,
    String? selectedEpisodeId,
  }) {
    return TimelineState(
      isLoading: isLoading ?? this.isLoading,
      episodes: episodes ?? this.episodes,
      errorMessage: errorMessage,
      selectedEpisodeId: selectedEpisodeId ?? this.selectedEpisodeId,
    );
  }
}

class TimelineNotifier extends Notifier<TimelineState> {
  @override
  TimelineState build() => const TimelineState();

  Future<void> loadEpisodes() async {
    state = state.copyWith(isLoading: true, errorMessage: null);
    try {
      final apiClient = ref.read(apiClientProvider);
      final list = await apiClient.fetchEpisodes();
      state = state.copyWith(isLoading: false, episodes: list);
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: 'Unable to connect to Project N daemon on local network.',
      );
    }
  }

  void selectEpisode(String id) {
    state = state.copyWith(selectedEpisodeId: id);
  }
}

final timelineProvider = NotifierProvider<TimelineNotifier, TimelineState>(
  TimelineNotifier.new,
);
