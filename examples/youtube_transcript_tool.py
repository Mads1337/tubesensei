from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
import re
import asyncio
import concurrent.futures
import logging

logger = logging.getLogger(__name__)


class YouTubeTranscriptToolInput(BaseModel):
    """Input schema for YouTubeTranscriptTool."""
    youtube_url: str = Field(..., description="YouTube URL to extract transcript from")


class BatchYouTubeTranscriptToolInput(BaseModel):
    """Input schema for batch YouTube transcript extraction."""
    youtube_urls: List[str] = Field(..., description="List of YouTube URLs to extract transcripts from")
    max_concurrent: int = Field(default=5, description="Maximum number of concurrent transcript extractions")


class YouTubeTranscriptTool(BaseTool):
    name: str = "YouTube Transcript Extractor"
    description: str = (
        "Extracts the transcript from a YouTube video URL. Provide a valid YouTube URL "
        "and this tool will return the full transcript text of the video."
    )
    args_schema: Type[BaseModel] = YouTubeTranscriptToolInput

    def _run(self, youtube_url: str) -> str:
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(youtube_url)
            if not video_id:
                return "Error: Could not extract video ID from the provided URL"
            
            # Get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Format transcript into readable text
            formatted_transcript = self._format_transcript(transcript)
            
            return formatted_transcript
            
        except Exception as e:
            return f"Error retrieving transcript: {str(e)}"
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _format_transcript(self, transcript: list) -> str:
        """Format transcript list into readable text"""
        formatted_text = ""
        for entry in transcript:
            text = entry['text'].replace('\n', ' ')
            formatted_text += f"{text} "
        
        return formatted_text.strip()
    
    async def _extract_transcript_async(self, video_id: str) -> Dict[str, Any]:
        """Extract transcript for a single video ID asynchronously."""
        try:
            # Run the blocking transcript extraction in a thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                transcript = await loop.run_in_executor(
                    executor, 
                    YouTubeTranscriptApi.get_transcript, 
                    video_id
                )
            
            formatted_transcript = self._format_transcript(transcript)
            return {
                'video_id': video_id,
                'transcript': formatted_transcript,
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'video_id': video_id,
                'transcript': None,
                'success': False,
                'error': str(e)
            }


class BatchYouTubeTranscriptTool(BaseTool):
    """
    Batch YouTube Transcript Tool for concurrent transcript extraction.
    
    This tool allows extracting transcripts from multiple YouTube videos
    concurrently, significantly reducing the total time needed for batch operations.
    """
    
    name: str = "Batch YouTube Transcript Extractor"
    description: str = (
        "Extracts transcripts from multiple YouTube videos concurrently. "
        "Provide a list of YouTube URLs and this tool will process them in parallel, "
        "returning organized results for each video. Much faster than sequential extraction."
    )
    args_schema: Type[BaseModel] = BatchYouTubeTranscriptToolInput
    
    def _run(self, youtube_urls: List[str], max_concurrent: int = 5) -> str:
        """
        Extract transcripts from multiple YouTube URLs concurrently.
        
        Args:
            youtube_urls: List of YouTube URLs to process
            max_concurrent: Maximum number of concurrent extractions
            
        Returns:
            Formatted string containing all transcript results
        """
        if not youtube_urls:
            return "Error: No YouTube URLs provided."
        
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(youtube_urls))
        
        try:
            # Run async transcript extraction
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    self._batch_extract_transcripts(unique_urls, max_concurrent)
                )
                
                # Format and return results
                return self._format_batch_results(results)
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error in batch transcript extraction: {str(e)}")
            return f"Error performing batch transcript extraction: {str(e)}"
    
    async def _batch_extract_transcripts(
        self, 
        urls: List[str], 
        max_concurrent: int
    ) -> List[Dict[str, Any]]:
        """Extract transcripts from multiple URLs with semaphore-based rate limiting."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                video_id = self._extract_video_id(url)
                if not video_id:
                    return {
                        'url': url,
                        'video_id': None,
                        'transcript': None,
                        'success': False,
                        'error': 'Could not extract video ID from URL'
                    }
                
                result = await self._extract_transcript_async(video_id)
                result['url'] = url
                return result
        
        # Create tasks for all URLs
        tasks = [extract_with_semaphore(url) for url in urls]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'url': urls[i],
                    'video_id': None,
                    'transcript': None,
                    'success': False,
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _extract_transcript_async(self, video_id: str) -> Dict[str, Any]:
        """Extract transcript for a single video ID asynchronously."""
        try:
            # Run the blocking transcript extraction in a thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                transcript = await loop.run_in_executor(
                    executor, 
                    YouTubeTranscriptApi.get_transcript, 
                    video_id
                )
            
            formatted_transcript = self._format_transcript(transcript)
            return {
                'video_id': video_id,
                'transcript': formatted_transcript,
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'video_id': video_id,
                'transcript': None,
                'success': False,
                'error': str(e)
            }
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _format_transcript(self, transcript: list) -> str:
        """Format transcript list into readable text"""
        formatted_text = ""
        for entry in transcript:
            text = entry['text'].replace('\n', ' ')
            formatted_text += f"{text} "
        
        return formatted_text.strip()
    
    def _format_batch_results(self, results: List[Dict[str, Any]]) -> str:
        """Format batch transcript results into a readable string."""
        if not results:
            return "No results to display."
        
        formatted = []
        successful_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - successful_count
        
        # Add summary
        formatted.append("BATCH YOUTUBE TRANSCRIPT EXTRACTION RESULTS")
        formatted.append("=" * 55)
        formatted.append(f"Total Videos: {len(results)}")
        formatted.append(f"Successful Extractions: {successful_count}")
        formatted.append(f"Failed Extractions: {failed_count}")
        formatted.append("")
        
        # Add individual results
        for i, result in enumerate(results, 1):
            formatted.append(f"=== VIDEO {i} ===")
            formatted.append(f"URL: {result['url']}")
            formatted.append(f"Video ID: {result.get('video_id', 'N/A')}")
            formatted.append(f"Status: {'✅ Success' if result['success'] else '❌ Failed'}")
            
            if result['success']:
                transcript = result['transcript']
                word_count = len(transcript.split()) if transcript else 0
                formatted.append(f"Word Count: {word_count}")
                formatted.append("")
                formatted.append("TRANSCRIPT:")
                formatted.append(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
            else:
                formatted.append(f"Error: {result['error']}")
            
            formatted.append("")
        
        return "\n".join(formatted)