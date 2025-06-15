"use client"; // Required for App Router client components

import * as React from "react";
import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card"; // Adjust path if needed
import { Button } from "@/components/ui/button"; // Adjust path if needed
import {
  ListVideo,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  Hourglass,
  Search,
  Zap,
  Upload as UploadIcon, // Renamed to avoid conflict with lucide-react's Upload
  Download, // For download button later
  Eye, // For view details/clip later
} from "lucide-react";

// Interface for the video data matching your backend's VideoListItemResponse
interface VideoItem {
  video_uuid: string;
  original_filename_server?: string | null;
  processing_status: string;
  created_at: string; // Comes as string from JSON, will be parsed
  updated_at?: string | null; // Optional
  highlight_clip_path?: string | null;
  download_url?: string | null; // Will be constructed later
}

// For status polling individual videos
interface VideoStatusUpdate {
  video_uuid: string;
  processing_status: string;
  error_message?: string | null;
  highlight_clip_path?: string | null;
  download_url?: string | null; // If backend provides this directly after generation
}

export default function DashboardPage() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isPolling, setIsPolling] = useState<boolean>(false); // For polling indicator
  const [error, setError] = useState<string | null>(null);

  // Ref to hold the latest videos state for use inside setInterval
  // This helps avoid stale closure issues with the videos state in the interval callback.
  const videosRef = useRef(videos);
  useEffect(() => {
    videosRef.current = videos;
  }, [videos]);

  const fetchVideos = useCallback(async (isInitialLoad = false) => {
    if (isInitialLoad) {
      setIsLoading(true); // Only show full page loader on initial load
    }
    setError(null);
    try {
      const response = await fetch("http://localhost:8001/videos"); // Ensure backend is running
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          detail: "Server error fetching videos.",
        }));
        throw new Error(
          errorData.detail || `Failed to fetch videos: ${response.status}`
        );
      }
      const data: VideoItem[] = await response.json();
      setVideos(data);
    } catch (err: any) {
      setError(
        err.message || "An unknown error occurred while fetching videos."
      );
      setVideos([]); // Clear videos on error
    } finally {
      if (isInitialLoad) {
        setIsLoading(false);
      }
    }
  }, []);

  // Effect for initial load
  useEffect(() => {
    fetchVideos(true);
  }, [fetchVideos]);

  // Effect for polling statuses
  useEffect(() => {
    const pollInterval = 7000; // Poll every 7 seconds
    let intervalId: NodeJS.Timeout;

    const pollVideoStatuses = async () => {
      const currentVideos = videosRef.current; // Use the ref to get the latest videos state
      const videosToPoll = currentVideos.filter(
        (v) =>
          !(
            v.processing_status.toLowerCase().includes("ready") || // READY_FOR_SEARCH
            v.processing_status.toLowerCase().includes("failed") || // PROCESSING_FAILED, PARTIAL_FAILURE
            v.processing_status.toLowerCase().includes("generated") // HIGHLIGHT_GENERATED
          )
      );

      if (videosToPoll.length > 0) {
        if (!isPolling) setIsPolling(true); // Indicate polling is active
        console.log(`Polling statuses for ${videosToPoll.length} videos...`);
        let madeAnUpdate = false;
        const updatedVideosArray = [...currentVideos]; // Create a mutable copy

        for (const videoToPoll of videosToPoll) {
          try {
            const response = await fetch(
              `http://localhost:8001/videos/${videoToPoll.video_uuid}/status`
            );
            if (response.ok) {
              const statusData: VideoStatusUpdate = await response.json();
              const videoIndex = updatedVideosArray.findIndex(
                (v) => v.video_uuid === statusData.video_uuid
              );
              if (videoIndex !== -1 && updatedVideosArray[videoIndex].processing_status !== statusData.processing_status) {
                console.log(`Status update for ${statusData.video_uuid}: ${updatedVideosArray[videoIndex].processing_status} -> ${statusData.processing_status}`);
                updatedVideosArray[videoIndex] = {
                  ...updatedVideosArray[videoIndex],
                  processing_status: statusData.processing_status,
                  highlight_clip_path: statusData.highlight_clip_path, // Update this too
                  // error_message: statusData.error_message, // If you want to display errors
                };
                madeAnUpdate = true;
              }
            } else {
                console.warn(`Failed to poll status for ${videoToPoll.video_uuid}: ${response.status}`);
            }
          } catch (pollError) {
            console.error(
              `Error polling status for ${videoToPoll.video_uuid}:`,
              pollError
            );
          }
        }
        if (madeAnUpdate) {
          setVideos(updatedVideosArray);
        }
      } else {
        if (isPolling) setIsPolling(false); // No more videos to poll actively
      }
    };

    intervalId = setInterval(pollVideoStatuses, pollInterval);

    return () => clearInterval(intervalId); // Cleanup interval on unmount
  }, [videos, isPolling]); // Rerun when videos change to reassess if polling is needed, or when isPolling changes

  const getStatusInfo = (
    status: string
  ): { text: string; color: string; icon: React.ElementType } => {
    const s = status.toLowerCase();
    if (s.includes("failed")) return { text: "Failed", color: "text-red-600 dark:text-red-400", icon: AlertTriangle };
    if (s.includes("ready_for_search")) return { text: "Ready for Search", color: "text-green-600 dark:text-green-400", icon: CheckCircle2 };
    if (s.includes("highlight_generated")) return { text: "Highlight Ready", color: "text-emerald-600 dark:text-emerald-400", icon: Zap };
    if (s.includes("processing") || s.includes("extracting") || s.includes("transcribing") || s.includes("embedding") || s.includes("generating")) {
        return { text: "Processing...", color: "text-blue-600 dark:text-blue-400", icon: Hourglass };
    }
    if (s.includes("uploaded")) return {text: "Uploaded", color: "text-sky-600 dark:text-sky-400", icon: UploadIcon }
    return { text: status.replace(/_/g, " ").toUpperCase(), color: "text-gray-600 dark:text-gray-400", icon: ListVideo };
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8 bg-gray-100 dark:bg-gray-900">
      <div className="w-full max-w-5xl">
        <div className="flex flex-col sm:flex-row justify-between items-center mb-8 gap-4">
          <h1 className="text-3xl font-bold tracking-tight dark:text-slate-100">
            ClipPilot Dashboard
          </h1>
          <div className="flex items-center gap-2">
            {isPolling && <RefreshCw className="h-5 w-5 text-blue-500 animate-spin" />}
            <Button
              onClick={() => fetchVideos(true)} // Pass true for initialLoad behavior on manual refresh
              variant="outline"
              size="sm"
              disabled={isLoading}
            >
              <RefreshCw
                className={`mr-2 h-4 w-4 ${
                  isLoading && !isPolling ? "animate-spin" : "" // Only spin if it's a full load, not just polling
                }`}
              />
              Refresh List
            </Button>
            <Button asChild size="sm">
              <Link href="/">
                <UploadIcon className="mr-2 h-4 w-4" /> Upload New Video
              </Link>
            </Button>
          </div>
        </div>

        {isLoading && videos.length === 0 && ( // Show full page loader only if truly loading initial data
          <div className="text-center py-10">
            <RefreshCw className="mx-auto h-10 w-10 text-gray-400 animate-spin" />
            <p className="mt-3 text-gray-500 dark:text-gray-400">
              Loading videos...
            </p>
          </div>
        )}
        {error && (
          <div className="p-4 mb-6 text-sm text-red-700 bg-red-100 rounded-lg dark:bg-red-900/30 dark:text-red-300 flex items-center shadow">
            <AlertTriangle className="w-5 h-5 mr-3 flex-shrink-0" />
            <div>
              <p className="font-semibold">Error Fetching Videos:</p>
              <p>{error}</p>
            </div>
          </div>
        )}

        {!isLoading && !error && videos.length === 0 && (
          <div className="text-center py-20 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg">
            <ListVideo className="mx-auto h-16 w-16 text-gray-400" />
            <p className="mt-4 text-xl font-medium text-gray-500 dark:text-gray-400">
              No videos found.
            </p>
            <p className="mt-2 text-sm text-gray-400 dark:text-gray-500">
              Looks like your cockpit is empty! Upload a video to get started.
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {videos.map((video) => {
            const statusInfo = getStatusInfo(video.processing_status);
            const canSearch = video.processing_status.toLowerCase().includes("ready_for_search") || video.processing_status.toLowerCase().includes("highlight_generated");
            const canGenerate = video.processing_status.toLowerCase().includes("ready_for_search") || video.processing_status.toLowerCase().includes("highlight_generated");

            return (
              <Card
                key={video.video_uuid}
                className="flex flex-col shadow-lg hover:shadow-xl transition-shadow duration-300 dark:bg-slate-800 dark:border-slate-700"
              >
                <CardHeader className="pb-3">
                  <CardTitle
                    className="text-lg truncate dark:text-slate-100"
                    title={video.original_filename_server || video.video_uuid}
                  >
                    {video.original_filename_server || "Untitled Video"}
                  </CardTitle>
                  <CardDescription className="text-xs dark:text-slate-400">
                    ID:{" "}
                    <span className="font-mono">
                      {video.video_uuid.substring(0, 8)}...
                    </span>
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex-grow space-y-2">
                  <p className="text-xs text-gray-500 dark:text-slate-500">
                    Uploaded:{" "}
                    {new Date(video.created_at).toLocaleDateString()}{" "}
                    {new Date(video.created_at).toLocaleTimeString()}
                  </p>
                  <div
                    className={`flex items-center text-sm font-medium ${statusInfo.color}`}
                  >
                    <statusInfo.icon className={`w-4 h-4 mr-2 flex-shrink-0 ${statusInfo.text === "Processing..." ? "animate-spin" : ""}`} />
                    {statusInfo.text}
                  </div>
                </CardContent>
                <CardFooter className="border-t pt-4 dark:border-slate-700 flex-wrap gap-2">
                  {canSearch && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-1 min-w-[120px] dark:text-slate-300 dark:border-slate-600 dark:hover:bg-slate-700"
                      asChild
                    >
                      <Link href={`/search?video_uuid=${video.video_uuid}&filename=${encodeURIComponent(video.original_filename_server || video.video_uuid)}`}>
                        <Search className="mr-2 h-4 w-4" /> Search
                      </Link>
                    </Button>
                  )}
                  {canGenerate && (
                     <Button
                        variant="default"
                        size="sm"
                        className="flex-1 min-w-[120px]" // Ensure buttons can fit
                        asChild
                    >
                        {/* This link will go to a page dedicated to generating highlights for THIS video */}
                        <Link href={`/generate?video_uuid=${video.video_uuid}&filename=${encodeURIComponent(video.original_filename_server || video.video_uuid)}`}>
                            <Zap className="mr-2 h-4 w-4" /> Highlights
                        </Link>
                    </Button>
                  )}
                  {/* Placeholder for a view/download clip button if highlight_clip_path exists */}
                  {video.highlight_clip_path && video.processing_status.toLowerCase().includes("generated") && (
                     <Button variant="secondary" size="sm" className="flex-1 min-w-[120px]" asChild>
                        {/* This would link to your /clips/{video_uuid}/{filename} endpoint eventually */}
                        <Link href={`/placeholder-view-clip/${video.video_uuid}`}>
                            <Eye className="mr-2 h-4 w-4" /> View Clip
                        </Link>
                     </Button>
                  )}
                </CardFooter>
              </Card>
            );
          })}
        </div>
      </div>
    </main>
  );
}