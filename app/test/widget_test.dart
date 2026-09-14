import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:project_n_companion/main.dart';

void main() {
  testWidgets('Renders ProjectNApp shell with navigation destinations', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const ProviderScope(child: ProjectNApp()));

    expect(find.text('Timeline'), findsOneWidget);
    expect(find.text('Record'), findsOneWidget);
    expect(find.text('Triage'), findsOneWidget);
    expect(find.text('Pairing'), findsOneWidget);

    // Navigate to Record tab
    await tester.tap(find.text('Record'));
    await tester.pumpAndSettle();

    expect(find.text('Capture Observation'), findsOneWidget);
    expect(find.text('Start Recording (30–120s)'), findsOneWidget);

    // Navigate to Triage tab
    await tester.tap(find.text('Triage'));
    await tester.pumpAndSettle();

    expect(find.text('NCCPC-PV Pain Checklist'), findsOneWidget);
    expect(
      find.textContaining('Total Score: 0 / 11 Threshold'),
      findsOneWidget,
    );
  });
}
