import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:project_n_companion/core/contracts/vocabularies.dart';
import 'package:project_n_companion/features/triage/nccpc_provider.dart';

void main() {
  group('NCCPC Psychometrics Tests', () {
    test('Calculates score and triggers pain cutoff alert at >= 11', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);

      final notifier = container.read(nccpcProvider.notifier);

      // Below threshold: 2 + 3 = 5
      notifier.setItemScore('crying', 2);
      notifier.setItemScore('screaming_yelling', 3);
      expect(container.read(nccpcProvider).totalScore, 5);
      expect(container.read(nccpcProvider).painCutoffBreached, false);

      // Above threshold: 5 + 3 + 3 = 11
      notifier.setItemScore('stiff_spastic_rigid_tense', 3);
      notifier.setItemScore('protecting_favoring_guarding', 3);
      expect(container.read(nccpcProvider).totalScore, 11);
      expect(container.read(nccpcProvider).painCutoffBreached, true);
      expect(nccpcPvCutoffThreshold, 11);

      // Reset
      notifier.reset();
      expect(container.read(nccpcProvider).totalScore, 0);
      expect(container.read(nccpcProvider).painCutoffBreached, false);
    });
  });
}
