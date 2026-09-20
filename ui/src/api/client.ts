/**
 * Project N: Base HTTP API Client.
 * Strict offline loopback communication with error handling and auth token resolution.
 */

const API_BASE = "";

export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

export function getStoredAuthToken(): string | null {
  try {
    return localStorage.getItem("project_n_token");
  } catch {
    return null;
  }
}

export function setStoredAuthToken(token: string): void {
  try {
    localStorage.setItem("project_n_token", token);
  } catch {
    // Ignore storage failures in sandbox
  }
}

let tokenPromise: Promise<string | null> | null = null;

export async function ensureLocalToken(): Promise<string | null> {
  const existing = getStoredAuthToken();
  if (existing) return existing;

  if (tokenPromise) return tokenPromise;

  tokenPromise = (async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/local-token`, {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      if (res.ok) {
        const data = (await res.json()) as { token: string };
        setStoredAuthToken(data.token);
        return data.token;
      }
    } catch {
      // Offline or network error fallback
    } finally {
      tokenPromise = null;
    }
    return null;
  })();

  return tokenPromise;
}

export async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");

  if (!headers.has("Content-Type") && options.body && typeof options.body === "string") {
    headers.set("Content-Type", "application/json");
  }

  let token = getStoredAuthToken();
  if (!token && !endpoint.includes("/auth/")) {
    token = await ensureLocalToken();
  }
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorData: unknown;
    try {
      errorData = await response.json();
    } catch {
      errorData = await response.text();
    }
    const message =
      typeof errorData === "object" && errorData && "detail" in errorData
        ? String((errorData as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`;
    throw new ApiError(message, response.status, errorData);
  }

  return response.json() as Promise<T>;
}
