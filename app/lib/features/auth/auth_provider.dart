/// Project N: Authentication & Pairing Provider.
/// Manages companion pairing state and local device credential token.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/network/api_client.dart';

enum AuthStatus { unpaired, pairing, paired, error }

class AuthState {
  final AuthStatus status;
  final String? token;
  final String? errorMessage;
  final String deviceId;

  const AuthState({
    this.status = AuthStatus.unpaired,
    this.token,
    this.errorMessage,
    this.deviceId = 'caregiver_companion_device',
  });

  AuthState copyWith({
    AuthStatus? status,
    String? token,
    String? errorMessage,
  }) {
    return AuthState(
      status: status ?? this.status,
      token: token ?? this.token,
      errorMessage: errorMessage,
      deviceId: deviceId,
    );
  }
}

class AuthNotifier extends Notifier<AuthState> {
  @override
  AuthState build() => const AuthState();

  Future<bool> pairWithPin(String pin) async {
    state = state.copyWith(status: AuthStatus.pairing, errorMessage: null);
    try {
      final apiClient = ref.read(apiClientProvider);
      final token = await apiClient.pairDevice(state.deviceId, pin);
      state = state.copyWith(status: AuthStatus.paired, token: token);
      return true;
    } catch (e) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage:
            'Pairing failed: Please check the 6-digit PIN on your Mac.',
      );
      return false;
    }
  }

  void logout() {
    ref.read(apiClientProvider).setToken('');
    state = const AuthState(status: AuthStatus.unpaired);
  }
}

final apiClientProvider = Provider<ApiClient>((ref) {
  return ApiClient();
});

final authProvider = NotifierProvider<AuthNotifier, AuthState>(
  AuthNotifier.new,
);
