"use client";

import * as React from "react";
import { useState, useEffect, useMemo } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import { Search as SearchIcon, AlertTriangle, RefreshCw, Zap, ArrowLeft } from "lucide-react";

// --- INTERFACES & TYPES ---
interface BaseSearchResult {
  id: string; // ID must be unique
  video_uuid: string; // Made non-optional for consistency
  score?: number | null;
  score_type?: string | null;
  _raw_distance?: number | null;
}
interface TextSearchResult extends BaseSearchResult {
  segment_text: string;
  start_time: number;
  end_time: number;
}
interface VisionSearchResult extends BaseSearchResult {
  frame_filename: string;
  frame_timestamp_sec: number;
}
type SearchResultItem = TextSearchResult | VisionSearchResult;

interface HighlightSegmentRequestData {
  start_time: number;
  end_time: number;
  text_content: string;
  video_uuid: string;
}

// --- TYPE GUARDS ---
const isTextResult = (item: SearchResultItem): item is TextSearchResult => 'segment_text' in item;
const isVisionResult = (item: SearchResultItem): item is VisionSearchResult => 'frame_filename' in item;

export default function SearchPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // --- STATE MANAGEMENT ---
  const [queryText, setQueryText] = useState("");
  const [searchType, setSearchType] = useState<"text" | "visual">("text");
  const [targetVideoUuid, setTargetVideoUuid] = useState<string | null>(null);
  const [targetVideoFilename, setTargetVideoFilename] = useState<string | null>(null);
  const [topK, setTopK] = useState<number>(5);
  const [useRagRefinement, setUseRagRefinement] = useState<boolean>(false);

  const [searchResults, setSearchResults] = useState<SearchResultItem[] | null>(null);
  const [ragSummary, setRagSummary] = useState<string | null>(null);
  const [isLoadingSearch, setIsLoadingSearch] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // FIX: Use a Set of unique IDs for selection for simplicity and performance.
  const [selectedSegmentIds, setSelectedSegmentIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    const videoUuidFromUrl = searchParams.get("video_uuid");
    const filenameFromUrl = searchParams.get("filename");
    if (videoUuidFromUrl) setTargetVideoUuid(videoUuidFromUrl);
    if (filenameFromUrl) setTargetVideoFilename(filenameFromUrl);
  }, [searchParams]);

  const handleSearchSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoadingSearch(true);
    setSearchError(null);
    setSearchResults(null);
    setRagSummary(null);
    setSelectedSegmentIds(new Set()); // Reset selections on new search

    const requestBody: any = {
      query_text: queryText,
      search_type: searchType,
      top_k: topK,
      use_rag_refinement: useRagRefinement,
    };
    if (targetVideoUuid) requestBody.video_uuid = targetVideoUuid;

    try {
      const response = await fetch("http://localhost:8001/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Server error during search." }));
        throw new Error(errorData.detail || `Search failed: ${response.status}`);
      }
      const data = await response.json();
      setSearchResults(data.results || []);
      setRagSummary(data.rag_summary || null);
    } catch (err: any) {
      setSearchError(err.message || "An unknown error occurred during search.");
    } finally {
      setIsLoadingSearch(false);
    }
  };

  // FIX: Simplified selection handler using a Set of IDs.
  const handleSegmentSelectionChange = (segmentId: string, isSelected: boolean | "indeterminate") => {
    if (typeof isSelected !== 'boolean') return;

    setSelectedSegmentIds(prev => {
      const newSet = new Set(prev);
      if (isSelected) {
        newSet.add(segmentId);
      } else {
        newSet.delete(segmentId);
      }
      return newSet;
    });
  };
  
  // Memoize selected segments to avoid re-filtering on every render
  const selectedSegments = useMemo(() => {
    if (!searchResults) return [];
    return searchResults.filter(r => selectedSegmentIds.has(r.id));
  }, [selectedSegmentIds, searchResults]);


  const handleGenerateHighlightFromSelected = () => {
    if (selectedSegments.length === 0) {
      alert("Please select at least one segment to generate a highlight.");
      return;
    }

    // FIX: Safely determine the video UUID and ensure all segments are from the same video.
    const firstVideoUuid = selectedSegments[0].video_uuid;
    const allSameVideo = selectedSegments.every(s => s.video_uuid === firstVideoUuid);

    if (!allSameVideo) {
      alert("Error: Please select segments from only one video to generate a highlight.");
      return;
    }

    // Convert selected segments to the request format
    const segmentsForRequest: HighlightSegmentRequestData[] = selectedSegments.map(segment => {
      if (isTextResult(segment)) {
        return {
          start_time: segment.start_time,
          end_time: segment.end_time,
          text_content: segment.segment_text,
          video_uuid: segment.video_uuid,
        };
      } else { // isVisionResult
        return {
          start_time: segment.frame_timestamp_sec,
          end_time: segment.frame_timestamp_sec + 3.0, // Visual segments are 3s long by default
          text_content: `Visual: ${segment.frame_filename}`,
          video_uuid: segment.video_uuid,
        };
      }
    });

    const segmentsQueryParam = encodeURIComponent(JSON.stringify(segmentsForRequest));
    const filenameForUrl = encodeURIComponent(targetVideoFilename || firstVideoUuid);
    router.push(`/generate?video_uuid=${firstVideoUuid}&segments=${segmentsQueryParam}&filename=${filenameForUrl}`);
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8 bg-gray-100 dark:bg-gray-900">
      <div className="w-full max-w-4xl">
        <div className="mb-6">
            <Button variant="outline" size="sm" onClick={() => router.back()} className="mb-4">
                <ArrowLeft className="mr-2 h-4 w-4" /> Back
            </Button>
            <h1 className="text-3xl font-bold tracking-tight dark:text-slate-100">Search Video Content</h1>
            {targetVideoFilename && (
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                    Searching within: <strong>{targetVideoFilename}</strong> (ID: {targetVideoUuid?.substring(0,8)}...)
                </p>
            )}
        </div>

        <Card className="mb-8 dark:bg-slate-800 shadow-lg">
          <CardHeader>
            <CardTitle className="dark:text-slate-100">Search Parameters</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSearchSubmit} className="space-y-4">
              <div>
                <Label htmlFor="queryText" className="dark:text-slate-300">Search Query</Label>
                <Input
                  id="queryText"
                  type="text"
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  placeholder="e.g., customer mentioned 'thank you'"
                  required
                  className="dark:bg-slate-700 dark:text-slate-100 dark:border-slate-600"
                />
              </div>
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="flex-1">
                  <Label className="dark:text-slate-300">Search Type</Label>
                  <RadioGroup
                    value={searchType}
                    onValueChange={(value) => setSearchType(value as "text" | "visual")}
                    className="flex gap-4 mt-1"
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="text" id="searchText" className="dark:border-slate-600 dark:data-[state=checked]:bg-primary"/>
                      <Label htmlFor="searchText" className="dark:text-slate-300">Text</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="visual" id="searchVisual" className="dark:border-slate-600 dark:data-[state=checked]:bg-primary"/>
                      <Label htmlFor="searchVisual" className="dark:text-slate-300">Visual</Label>
                    </div>
                  </RadioGroup>
                </div>
                <div className="flex-1">
                  <Label htmlFor="topK" className="dark:text-slate-300">Top K Results</Label>
                  <Input
                    id="topK"
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value, 10) || 1))}
                    min="1" max="20"
                    className="dark:bg-slate-700 dark:text-slate-100 dark:border-slate-600"
                  />
                </div>
              </div>
              <div className="flex items-center space-x-2 pt-2">
                <Switch 
                    id="useRagRefinement" 
                    checked={useRagRefinement} 
                    onCheckedChange={setUseRagRefinement}
                    className="dark:data-[state=checked]:bg-primary"
                />
                <Label htmlFor="useRagRefinement" className="dark:text-slate-300">Use AI to summarize/refine search results (RAG)</Label>
              </div>
              <Button type="submit" disabled={isLoadingSearch || !queryText.trim()} className="w-full sm:w-auto">
                {isLoadingSearch ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : <SearchIcon className="mr-2 h-4 w-4" />}
                Search
              </Button>
            </form>
          </CardContent>
        </Card>

        {isLoadingSearch && (
            <div className="text-center py-10">
                <RefreshCw className="mx-auto h-8 w-8 text-gray-400 animate-spin" />
                <p className="mt-2 text-gray-500 dark:text-gray-400">Searching...</p>
            </div>
        )}
        {searchError && (
          <div className="p-4 mb-4 text-sm text-red-700 bg-red-100 rounded-lg dark:bg-red-900/30 dark:text-red-300 flex items-center shadow">
            <AlertTriangle className="w-5 h-5 mr-3 flex-shrink-0" />
            <div><p className="font-semibold">Search Error:</p><p>{searchError}</p></div>
          </div>
        )}
        
        {ragSummary && (
            <Card className="my-6 dark:bg-slate-800 shadow-lg">
                <CardHeader><CardTitle className="text-lg dark:text-slate-100">AI Summary of Search Results</CardTitle></CardHeader>
                <CardContent><p className="text-sm text-slate-600 dark:text-slate-300 whitespace-pre-wrap">{ragSummary}</p></CardContent>
            </Card>
        )}

        {searchResults && searchResults.length > 0 && (
          <div className="space-y-4">
            <div className="flex justify-between items-center mt-8 mb-4">
                <h2 className="text-2xl font-semibold dark:text-slate-100">Search Results ({searchResults.length})</h2>
                {selectedSegments.length > 0 && (
                    <Button onClick={handleGenerateHighlightFromSelected} size="sm">
                        <Zap className="mr-2 h-4 w-4"/> Generate Highlight ({selectedSegments.length} selected)
                    </Button>
                )}
            </div>
            {searchResults.map((result, index) => (
              <Card key={result.id} className="dark:bg-slate-800 shadow">
                <CardHeader className="flex flex-row justify-between items-start">
                  <div>
                    {/* FIX: Use type guard to determine title safely */}
                    <CardTitle className="text-md dark:text-slate-200">
                      {isTextResult(result) ? `Text Segment ${index + 1}` : `Visual Match ${index + 1}`}
                    </CardTitle>
                    {!targetVideoUuid && ( // Show source video only in global search results
                        <CardDescription className="text-xs dark:text-slate-400">From video: {result.video_uuid.substring(0,8)}...</CardDescription>
                    )}
                  </div>
                  <div className="flex items-center space-x-2">
                    <Label htmlFor={`select-segment-${result.id}`} className="text-xs dark:text-slate-400">Select</Label>
                    <Checkbox 
                        id={`select-segment-${result.id}`}
                        // FIX: Simplified onCheckedChange and checked logic
                        onCheckedChange={(checked) => handleSegmentSelectionChange(result.id, checked)}
                        checked={selectedSegmentIds.has(result.id)}
                        className="dark:border-slate-600 dark:data-[state=checked]:bg-primary"
                    />
                  </div>
                </CardHeader>
                <CardContent className="text-sm">
                  {/* FIX: Use type guards to safely render result content */}
                  {isTextResult(result) && (
                    <p className="mb-1 text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-700/50 p-2 rounded">"{result.segment_text}"</p>
                  )}
                  {isVisionResult(result) && (
                    <p className="mb-1 text-slate-700 dark:text-slate-300">Frame: {result.frame_filename}</p>
                  )}
                  <div className="text-xs text-slate-500 dark:text-slate-400 space-x-3">
                    <span>Score: {(result.score ?? 0).toFixed(3)} ({result.score_type || 'N/A'})</span>
                    {isTextResult(result) && <span>Time: {result.start_time.toFixed(2)}s - {result.end_time.toFixed(2)}s</span>}
                    {isVisionResult(result) && <span>Time: {result.frame_timestamp_sec.toFixed(2)}s</span>}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
        {searchResults && searchResults.length === 0 && !isLoadingSearch && (
            <p className="text-center text-gray-500 dark:text-gray-400 mt-8">No results found for your query.</p>
        )}
      </div>
    </main>
  );
}