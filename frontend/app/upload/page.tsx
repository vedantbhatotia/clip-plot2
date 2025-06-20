// app/(dashboard)/upload/page.tsx
"use client";

import * as React from "react";
import { useState } from "react";
import { useRouter } from "next/navigation"; // For redirecting after upload
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input"; // For file input
import { Label } from "@/components/ui/label";
import { AlertTriangle, CheckCircle2, Loader2, UploadCloud, ArrowLeft } from "lucide-react";

export default function UploadPage() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "success" | "error">("idle");
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setUploadStatus("idle"); // Reset status if a new file is selected
      setUploadMessage(null);
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFile) {
      setUploadMessage("Please select a video file to upload.");
      setUploadStatus("error");
      return;
    }

    setIsUploading(true);
    setUploadStatus("idle");
    setUploadMessage("Uploading video...");

    const formData = new FormData();
    formData.append("file", selectedFile); // "file" must match your FastAPI endpoint's File(...) parameter name

    try {
      const response = await fetch("http://localhost:8001/upload", {
        method: "POST",
        body: formData,
        // Headers are automatically set by browser for FormData with files
      });

      const responseData = await response.json();

      if (!response.ok) {
        // Try to get detail from FastAPI's HTTPException
        const errorMsg = responseData.detail || `Upload failed: ${response.status} ${response.statusText}`;
        throw new Error(errorMsg);
      }
      
      setUploadStatus("success");
      setUploadMessage(`Video '${selectedFile.name}' uploaded successfully! Processing started. Video ID: ${responseData.video_id}`);
      setSelectedFile(null); // Clear the file input
      // Optionally, redirect to dashboard after a delay or on button click
      setTimeout(() => {
        router.push("/dashboard"); // Navigate to videos dashboard
      }, 2000);

    } catch (err: any) {
      setUploadStatus("error");
      setUploadMessage(err.message || "An unknown error occurred during upload.");
      console.error("Upload error:", err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8 bg-gray-100 dark:bg-gray-900">
      <div className="w-full max-w-xl">
        <Button variant="outline" size="sm" onClick={() => router.back()} className="mb-6">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
        </Button>
        <Card className="dark:bg-slate-800 shadow-xl">
          <CardHeader>
            <CardTitle className="text-2xl dark:text-slate-100">Upload New Video</CardTitle>
            <CardDescription className="dark:text-slate-400">
              Select a video file from your computer to process with ClipPilot.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <Label htmlFor="videoFile" className="dark:text-slate-300">Video File</Label>
                <Input
                  id="videoFile"
                  type="file"
                  accept="video/*" // Accept all video types
                  onChange={handleFileChange}
                  required
                  className="mt-1 dark:bg-slate-700 dark:text-slate-100 dark:border-slate-600 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                />
                {selectedFile && (
                  <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
                    Selected: {selectedFile.name} ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
                  </p>
                )}
              </div>

              <Button type="submit" disabled={isUploading || !selectedFile} className="w-full">
                {isUploading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <UploadCloud className="mr-2 h-4 w-4" />
                )}
                {isUploading ? "Uploading..." : "Upload and Process"}
              </Button>
            </form>

            {uploadMessage && (
              <div className={`mt-4 p-3 rounded-md text-sm flex items-center ${
                uploadStatus === "success" ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300" 
                : uploadStatus === "error" ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300" 
                : "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300" 
              }`}>
                {uploadStatus === "success" && <CheckCircle2 className="w-5 h-5 mr-2 flex-shrink-0" />}
                {uploadStatus === "error" && <AlertTriangle className="w-5 h-5 mr-2 flex-shrink-0" />}
                {isUploading && uploadStatus === "idle" && <Loader2 className="w-5 h-5 mr-2 flex-shrink-0 animate-spin" />}
                {uploadMessage}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}