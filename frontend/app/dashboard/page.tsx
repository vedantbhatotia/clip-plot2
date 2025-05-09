"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, CheckCircle, AlertCircle } from "lucide-react"

export default function VideoUploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<"idle" | "success" | "error">("idle")
  const [errorMessage, setErrorMessage] = useState("")

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null

    if (selectedFile) {
      // Check if the file is a video
      if (!selectedFile.type.startsWith("video/")) {
        setErrorMessage("Please select a valid video file")
        setFile(null)
        return
      }

      setFile(selectedFile)
      setErrorMessage("")
      setUploadStatus("idle")
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!file) {
      setErrorMessage("Please select a video file")
      return
    }

    setIsUploading(true)
    setUploadStatus("idle")
    setErrorMessage("")

    try {
      const formData = new FormData()
      formData.append("file", file)

      console.log("Uploading file:", file.name, "Size:", (file.size / 1024 / 1024).toFixed(2) + "MB")

      const response = await fetch("http://localhost:8001/upload", {
        method: "POST",
        body: formData
      })

      if (response.ok) { 
        const data = await response.json() 
        console.log("File uploaded successfully to /upload endpoint. Backend response:", data)
        setUploadStatus("success")
        
      } else {
        
        const errorData = await response.json().catch(() => ({ detail: "Failed to parse error response from server." }))
        console.error("Server error uploading file:", response.status, errorData)
        setUploadStatus("error")
        setErrorMessage(errorData.detail || `Server error: ${response.status}`)
      }

    } catch (error) {
      console.error("Error uploading file:", error)
      setUploadStatus("error")
      setErrorMessage("Failed to upload video. Please try again.")
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-50">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Upload Video</CardTitle>
          <CardDescription>Select a video file to upload to our platform</CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent>
            <div className="grid gap-6">
              <div className="grid gap-2">
                <label
                  htmlFor="video-upload"
                  className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-md cursor-pointer border-gray-300 hover:border-primary transition-colors"
                >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <Upload className="w-8 h-8 mb-2 text-gray-500" />
                    <p className="mb-2 text-sm text-gray-500">
                      <span className="font-semibold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-xs text-gray-500">MP4, MOV, AVI, WebM (MAX. 100MB)</p>
                  </div>
                  <input
                    id="video-upload"
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleFileChange}
                    disabled={isUploading}
                  />
                </label>

                {file && (
                  <div className="text-sm text-gray-600 mt-2">
                    Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                  </div>
                )}

                {errorMessage && (
                  <div className="flex items-center text-sm text-red-500 mt-2">
                    <AlertCircle className="w-4 h-4 mr-1" />
                    {errorMessage}
                  </div>
                )}

                {uploadStatus === "success" && (
                  <div className="flex items-center text-sm text-green-500 mt-2">
                    <CheckCircle className="w-4 h-4 mr-1" />
                    Video uploaded successfully!
                  </div>
                )}
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button type="submit" className="w-full" disabled={!file || isUploading}>
              {isUploading ? (
                <>
                  <svg
                    className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Uploading...
                </>
              ) : (
                "Upload Video"
              )}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </main>
  )
}
