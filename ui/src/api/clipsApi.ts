import { request } from "./client";

export interface AnalyzeClipParams {
  antecedent_id: string;
  antecedent_notes?: string;
  caregiver_hypothesis?: string;
  setting?: string;
  observer?: string;
  captured_at?: string;
  generate_render?: boolean;
}

export interface AnalyzeClipResponse {
  task_id: string;
  stream_url: string;
}

export interface UploadInitResponse {
  upload_id: string;
  chunk_size: number;
}

export interface FinalizeResponse {
  clip_id: string;
  sha256: string;
  status: string;
}

export interface UploadProgress {
  stage: "hashing" | "uploading" | "finalizing" | "analyzing" | "completed";
  percent: number;
  uploadedBytes: number;
  totalBytes: number;
  currentChunk: number;
  totalChunks: number;
}

export async function computeFileSha256(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function triggerClipAnalysis(
  clipId: string,
  params: AnalyzeClipParams
): Promise<AnalyzeClipResponse> {
  return request<AnalyzeClipResponse>(`/api/v1/clips/${encodeURIComponent(clipId)}/analyze`, {
    method: "POST",
    body: JSON.stringify(params),
  });
}

const CHUNK_SIZE = 1048576; // 1 MB chunk default

export async function uploadAndAnalyzeClip(
  file: File,
  params: AnalyzeClipParams,
  onProgress?: (progress: UploadProgress) => void
): Promise<{ clipId: string; taskId: string }> {
  const totalBytes = file.size;
  const totalChunks = Math.max(1, Math.ceil(totalBytes / CHUNK_SIZE));

  // 1. Calculate SHA-256 Checksum
  onProgress?.({
    stage: "hashing",
    percent: 5,
    uploadedBytes: 0,
    totalBytes,
    currentChunk: 0,
    totalChunks,
  });
  const sha256 = await computeFileSha256(file);

  // 2. Init upload session
  const initRes = await request<UploadInitResponse>("/api/v1/clips/upload/init", {
    method: "POST",
    body: JSON.stringify({
      file_name: file.name,
      file_size: totalBytes,
      sha256,
      total_chunks: totalChunks,
    }),
  });
  const uploadId = initRes.upload_id;

  // 3. Upload binary chunks sequentially
  for (let idx = 0; idx < totalChunks; idx++) {
    const start = idx * CHUNK_SIZE;
    const end = Math.min(totalBytes, start + CHUNK_SIZE);
    const chunkBlob = file.slice(start, end);

    await request<{ chunk_index: number; received: boolean }>(
      `/api/v1/clips/upload/${encodeURIComponent(uploadId)}/chunk/${idx}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/octet-stream" },
        body: chunkBlob,
      }
    );

    const uploaded = end;
    const chunkPercent = 10 + Math.round((uploaded / totalBytes) * 70); // 10% - 80%
    onProgress?.({
      stage: "uploading",
      percent: chunkPercent,
      uploadedBytes: uploaded,
      totalBytes,
      currentChunk: idx + 1,
      totalChunks,
    });
  }

  // 4. Finalize upload (reassembles and vaults with AES-256-GCM)
  onProgress?.({
    stage: "finalizing",
    percent: 85,
    uploadedBytes: totalBytes,
    totalBytes,
    currentChunk: totalChunks,
    totalChunks,
  });

  const finalizeRes = await request<FinalizeResponse>(
    `/api/v1/clips/upload/${encodeURIComponent(uploadId)}/finalize`,
    {
      method: "POST",
      body: JSON.stringify({ metadata_json: { file_name: file.name } }),
    }
  );
  const clipId = finalizeRes.clip_id;

  // 5. Trigger multimodal analysis
  onProgress?.({
    stage: "analyzing",
    percent: 95,
    uploadedBytes: totalBytes,
    totalBytes,
    currentChunk: totalChunks,
    totalChunks,
  });

  const analyzeRes = await triggerClipAnalysis(clipId, params);

  onProgress?.({
    stage: "completed",
    percent: 100,
    uploadedBytes: totalBytes,
    totalBytes,
    currentChunk: totalChunks,
    totalChunks,
  });

  return { clipId, taskId: analyzeRes.task_id };
}
