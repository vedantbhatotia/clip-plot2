"use client";
import * as React from "react";
import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { AlertTriangle, CheckCircle, Clock, Eye, Download, Hourglass, Search as SearchIcon, UploadCloud } from "lucide-react";

// Interface for the full video data from the initial /videos list
interface VideoItem {
  video_uuid: string;
  original_filename_server?: string | null;
  processing_status: string;
  created_at: string;
  updated_at?: string | null;
  highlight_clip_path?: string | null; // The key in Supabase Storage
  download_url?: string | null;      // The signed URL for download
}

// Interface for the polled status update from /videos/{id}/status
interface VideoStatusUpdate {
  video_uuid: string;
  processing_status: string;
  error_message?: string | null;
  highlight_clip_path?: string | null;
  download_url?: string | null;
}

export default function DashboardPage() {
  const router = useRouter();
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Ref to hold the current videos to avoid stale state in the polling interval
  const videosRef = useRef(videos);
  useEffect(() => {
    videosRef.current = videos;
  }, [videos]);

  const fetchVideos = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8001/videos");
      if (!response.ok) throw new Error("Failed to fetch video list from the server.");
      const data: VideoItem[] = await response.json();
      setVideos(data);
    } catch (err: any) {
      setError(err.message || "An unknown error occurred while fetching videos.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchVideos(); // Initial fetch on component mount
  }, [fetchVideos]);

  // Polling logic for videos that are not in a final state.
  // This effect reacts to changes in the `videos` array itself.
  useEffect(() => {
    // Determine if any video is still in a processing state
    const shouldPoll = videos.some(v =>
      !['READY_FOR_SEARCH', 'HIGHLIGHT_GENERATED', 'PROCESSING_FAILED', 'PARTIAL_FAILURE', 'ERROR'].includes(v.processing_status)
    );

    if (!shouldPoll) {
      return; // Stop if no videos need polling
    }

    const intervalId = setInterval(async () => {
      const videosToPoll = videosRef.current.filter(v =>
        !['READY_FOR_SEARCH', 'HIGHLIGHT_GENERATED', 'PROCESSING_FAILED', 'PARTIAL_FAILURE', 'ERROR'].includes(v.processing_status)
      );

      // Failsafe in case the list of videos to poll becomes empty between intervals
      if (videosToPoll.length === 0) {
        clearInterval(intervalId);
        return;
      }
      
      let madeAnUpdate = false;
      const updatedVideosArray = [...videosRef.current];

      for (const videoToPoll of videosToPoll) {
        try {
          const res = await fetch(`http://localhost:8001/videos/${videoToPoll.video_uuid}/status`);
          if (!res.ok) continue;

          const statusData: VideoStatusUpdate = await res.json();
          const videoIndex = updatedVideosArray.findIndex(v => v.video_uuid === statusData.video_uuid);

          // Update if status or download URL has changed
          if (videoIndex !== -1 &&
             (updatedVideosArray[videoIndex].processing_status !== statusData.processing_status ||
              updatedVideosArray[videoIndex].download_url !== statusData.download_url)
          ) {
            updatedVideosArray[videoIndex] = { ...updatedVideosArray[videoIndex], ...statusData };
            madeAnUpdate = true;
          }
        } catch (e) {
          console.error(`Failed to poll status for ${videoToPoll.video_uuid}`, e);
        }
      }

      if (madeAnUpdate) {
        setVideos(updatedVideosArray);
      }
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(intervalId); // Cleanup on unmount or when `videos` changes
  }, [videos]);

  // Helper to get display info based on status string from backend
  const getStatusInfo = (status: string) => {
    const s = status.toUpperCase(); // Standardize to uppercase for reliable matching
    if (s === "PROCESSING_FAILED" || s === "ERROR") return { text: "Processing Failed", icon: AlertTriangle, color: "text-red-500" };
    if (s === "PARTIAL_FAILURE") return { text: "Partial Failure", icon: AlertTriangle, color: "text-orange-500" };
    if (s === "HIGHLIGHT_GENERATED") return { text: "Highlight Ready", icon: CheckCircle, color: "text-green-500" };
    if (s === "HIGHLIGHT_GENERATING") return { text: "Generating Clip...", icon: Hourglass, color: "text-purple-500 animate-pulse" };
    if (s === "READY_FOR_SEARCH") return { text: "Ready for Search", icon: CheckCircle, color: "text-blue-500" };
    if (s.includes("PROCESSING")) return { text: "Processing...", icon: Clock, color: "text-yellow-500 animate-spin" };
    if (s === "UPLOADED") return { text: "Uploaded", icon: UploadCloud, color: "text-gray-500" };
    return { text: status, icon: Clock, color: "text-gray-400" };
  };

  if (isLoading) {
    return <div className="flex justify-center items-center min-h-screen">Loading dashboard...</div>;
  }

  if (error) {
    return <div className="flex justify-center items-center min-h-screen text-red-500">{error}</div>;
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8 bg-gray-100 dark:bg-gray-900">
      <div className="w-full max-w-6xl">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold tracking-tight dark:text-slate-100">Video Dashboard</h1>
          <Button onClick={() => router.push('/')}>
            <UploadCloud className="mr-2 h-4 w-4" /> Upload New Video
          </Button>
        </div>

        {videos.length === 0 ? (
          <p className="text-center text-gray-500 dark:text-gray-400 mt-10">No videos uploaded yet. Click "Upload New Video" to get started.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {videos.map((video) => {
              const statusInfo = getStatusInfo(video.processing_status);
              const isSearchable = ['READY_FOR_SEARCH', 'HIGHLIGHT_GENERATED'].includes(video.processing_status);
              const isHighlightReady = video.processing_status === 'HIGHLIGHT_GENERATED' && !!video.download_url;

              return (
                <Card key={video.video_uuid} className="dark:bg-slate-800 shadow-lg flex flex-col">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg truncate dark:text-slate-100" title={video.original_filename_server || video.video_uuid}>
                      {video.original_filename_server || "Untitled Video"}
                    </CardTitle>
                    <CardDescription className="text-xs dark:text-slate-400">
                      ID: <span className="font-mono">{video.video_uuid.substring(0, 8)}...</span>
                    </CardDescription>
                  </CardHeader>

                  <CardContent className="flex-grow space-y-2">
                    <p className="text-xs text-gray-500 dark:text-slate-500">
                      Uploaded: {new Date(video.created_at).toLocaleString()}
                    </p>
                    <div className={`flex items-center text-sm font-medium ${statusInfo.color}`}>
                      <statusInfo.icon className="w-4 h-4 mr-2 flex-shrink-0" />
                      {statusInfo.text}
                    </div>
                  </CardContent>

                  <CardFooter className="border-t pt-4 dark:border-slate-700 flex flex-col items-stretch gap-2">
                    <div className="flex gap-2 w-full">
                      {isSearchable && (
                        <Button variant="outline" size="sm" className="flex-1" asChild>
                          <Link href={`/search?video_uuid=${video.video_uuid}&filename=${encodeURIComponent(video.original_filename_server || '')}`}>
                            <SearchIcon className="mr-2 h-4 w-4" /> Search
                          </Link>
                        </Button>
                      )}
                      {isHighlightReady && (
                        <>
                          <Button variant="secondary" size="sm" className="flex-1" asChild>
                            <Link href={video.download_url!} target="_blank" rel="noopener noreferrer">
                              <Eye className="mr-2 h-4 w-4" /> View
                            </Link>
                          </Button>
                          <a
                             href={video.download_url!}
                             download={
                                video.original_filename_server
                                    ? `highlight_${video.original_filename_server.split('.').slice(0, -1).join('.')}.mp4`
                                    : `highlight_${video.video_uuid.substring(0, 8)}.mp4`
                             }
                             className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors h-9 px-3 bg-primary text-primary-foreground shadow hover:bg-primary/90 flex-1"
                          >
                            <Download className="mr-2 h-4 w-4" /> Download
                          </a>
                        </>
                      )}
                    </div>
                  </CardFooter>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}