/// Project N: Cross-Platform Mobile Companion Client (Flutter).
/// Caregiver-operated edge application for observation capture and co-regulatory scaffolding.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'features/auth/pairing_screen.dart';
import 'features/recording/recording_screen.dart';
import 'features/timeline/timeline_screen.dart';
import 'features/triage/nccpc_screen.dart';

void main() {
  runApp(const ProviderScope(child: ProjectNApp()));
}

class ProjectNApp extends StatelessWidget {
  const ProjectNApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Project N Companion',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF00695C), // Calming deep teal
          brightness: Brightness.light,
        ),
        appBarTheme: const AppBarTheme(centerTitle: true, elevation: 0),
      ),
      home: const MainNavigationShell(),
    );
  }
}

class MainNavigationShell extends StatefulWidget {
  const MainNavigationShell({super.key});

  @override
  State<MainNavigationShell> createState() => _MainNavigationShellState();
}

class _MainNavigationShellState extends State<MainNavigationShell> {
  int _currentIndex = 0;

  final List<Widget> _screens = const [
    TimelineScreen(),
    RecordingScreen(),
    NCCPCScreen(),
    PairingScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_currentIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (idx) => setState(() => _currentIndex = idx),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.timeline), label: 'Timeline'),
          NavigationDestination(icon: Icon(Icons.videocam), label: 'Record'),
          NavigationDestination(
            icon: Icon(Icons.health_and_safety_outlined),
            label: 'Triage',
          ),
          NavigationDestination(
            icon: Icon(Icons.phonelink_setup),
            label: 'Pairing',
          ),
        ],
      ),
    );
  }
}
