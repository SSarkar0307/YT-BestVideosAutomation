#!/usr/bin/env python3
"""
Enhanced YouTube Video Search Script with Text/Voice Input
Accepts text or voice input as the search query and maps to `python script.py -q "<query>" --force-refresh`.
"""

import os
import sys
import random
import time
import pickle
import hashlib
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from user_input import get_user_query

# Third-party imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import isodate
    from dotenv import load_dotenv
    from tabulate import tabulate
    import humanize
    from tqdm import tqdm
    from gemini_analyzer import GeminiAnalyzer
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages with: pip install google-api-python-client isodate python-dotenv tabulate humanize tqdm google-generativeai speechrecognition")
    sys.exit(1)

# === CONFIGURATION ===
DEFAULT_CONFIG = {
    'max_results': 20,
    'min_duration_min': 4,
    'max_duration_min': 20,
    'days_back': 14,
    'cache_time': 3600,  # 1 hour in seconds
    'max_pages': 3,      # Reduced to avoid quota issues
    'max_cache_size_mb': 50,
    'max_cache_age_days': 7,
    'api_key_env_var': 'YOUTUBE_API_KEY'
}

# === API HELPERS ===
def load_api_key() -> str:
    """Load YouTube API key from environment variables or .env file"""
    load_dotenv()
    api_key = os.environ.get(DEFAULT_CONFIG['api_key_env_var'])
    
    if not api_key:
        print("Error: YouTube API key not found.")
        print("Please set the YOUTUBE_API_KEY environment variable or add it to a .env file.")
        sys.exit(1)
        
    return api_key

def build_youtube_client(api_key: str):
    """Build and return a YouTube API client"""
    try:
        client = build('youtube', 'v3', developerKey=api_key)
        
        try:
            execute_with_retry(client.videos().list(
                part="snippet",
                id="dQw4w9WgXcQ"
            ))
            return client
        except HttpError as e:
            if e.resp.status == 400:
                print("Error: Invalid API key or permissions")
                print("1. Check key at https://console.cloud.google.com/apis/credentials")
                print("2. Ensure 'YouTube Data API v3' is enabled")
            elif e.resp.status == 403:
                print("Error: API access forbidden")
                print("Possible causes:")
                print("- Quota exceeded (check at https://console.cloud.google.com/iam-admin/quotas)")
                print("- API key restrictions (IP/HTTP referrers)")
            sys.exit(1)
    except Exception as e:
        print(f"Connection error: {str(e)}")
        print("Check your internet connection and try again")
        sys.exit(1)

def execute_with_retry(request, max_retries: int = 5, min_delay: float = 0.5):
    """Execute an API request with exponential backoff retry logic and minimum delay."""
    for retry in range(max_retries):
        try:
            response = request.execute()
            time.sleep(min_delay)
            return response
        except HttpError as e:
            with open("youtube_api_errors.log", "a") as f:
                f.write(f"{datetime.now()}: HTTP {e.resp.status} - {e.content.decode()}\n")
            if e.resp.status in [403, 429]:
                if retry == max_retries - 1:
                    raise
                wait_time = (2 ** retry) + random.random()
                print(f"API quota limit approached. Waiting {wait_time:.1f} seconds...")
                try:
                    time.sleep(wait_time)
                except KeyboardInterrupt:
                    if input("\nCancel operation? (y/n): ").lower() != 'y':
                        continue
                    raise
            else:
                raise
        except KeyboardInterrupt:
            if input("\nCancel operation? (y/n): ").lower() != 'y':
                continue
            raise

def estimate_quota_usage(pages: int) -> int:
    """Returns estimated quota units needed"""
    search_units = 100 * pages
    details_units = 1 * pages * 50
    return search_units + details_units

# === CACHING FUNCTIONS ===
def get_cache_key(query: str, days: int, min_duration: int, max_duration: int, 
                 min_views: Optional[int] = None, channels_blacklist: Optional[str] = None, 
                 title_blacklist: Optional[str] = None) -> str:
    """Generate a unique cache key based on search parameters"""
    key_string = f"{query}_{days}_{min_duration}_{max_duration}"
    
    if min_views:
        key_string += f"_views{min_views}"
    if channels_blacklist:
        key_string += f"_chbl{channels_blacklist}"
    if title_blacklist:
        key_string += f"_titlebl{title_blacklist}"
        
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cached_results(cache_key: str, cache_time: int) -> Optional[List[Dict[str, Any]]]:
    """Retrieve cached results if they exist and are not expired"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    if os.path.exists(cache_file):
        try:
            import json
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                cache_timestamp = cache_data.get('timestamp', 0)
                
                if datetime.now().timestamp() - cache_timestamp < cache_time:
                    print("Using cached results")
                    return cache_data.get('results')
        except Exception as e:
            print(f"Error reading cache: {e}")
    
    return None

def cache_results(cache_key: str, results: List[Dict[str, Any]]) -> None:
    """Save results to cache using JSON for better security and portability"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        import json
        with open(os.path.join(cache_dir, f"{cache_key}.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().timestamp(),
                'results': results
            }, f, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving to cache: {e}")

def clean_old_cache() -> None:
    """Remove cached files exceeding size/age limits"""
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    if not os.path.exists(cache_dir):
        return

    total_size = 0
    now = time.time()
    
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        try:
            file_stat = os.stat(filepath)
            if now - file_stat.st_mtime > DEFAULT_CONFIG['max_cache_age_days'] * 86400:
                os.unlink(filepath)
                continue
            total_size += file_stat.st_size
        except Exception:
            continue

    if total_size > DEFAULT_CONFIG['max_cache_size_mb'] * 1024 * 1024:
        files = sorted(
            [(os.path.join(cache_dir, f), os.stat(os.path.join(cache_dir, f)).st_mtime) 
             for f in os.listdir(cache_dir)],
            key=lambda x: x[1]
        )
        while total_size > DEFAULT_CONFIG['max_cache_size_mb'] * 1024 * 1024 and files:
            try:
                file_size = os.stat(files[0][0]).st_size
                os.unlink(files[0][0])
                total_size -= file_size
                files.pop(0)
            except Exception:
                files.pop(0)

# === SEARCH FUNCTIONS ===
def search_videos(youtube, query: str, published_after: str, max_pages: int = 3) -> List[str]:
    """Search for videos and return their IDs"""
    all_video_ids = []
    next_page_token = None
    
    progress_bar = tqdm(total=max_pages, desc="Searching videos", unit="page")
    
    for page in range(max_pages):
        try:
            search_args = {
                'q': query,
                'part': 'id,snippet',
                'maxResults': 50,
                'type': 'video',
                'order': 'relevance',
                'publishedAfter': published_after
            }
            
            if next_page_token:
                search_args['pageToken'] = next_page_token
            
            search_response = execute_with_retry(youtube.search().list(**search_args))
            
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            all_video_ids.extend(video_ids)
            
            progress_bar.update(1)
            progress_bar.set_postfix({"Found": len(all_video_ids)})
            
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
                
            time.sleep(0.5)
        except Exception as e:
            progress_bar.write(f"Error fetching search results on page {page+1}: {e}")
            break
    
    progress_bar.close()
    return all_video_ids

def get_video_details(youtube, video_ids: List[str]) -> Dict[str, Any]:
    """Get detailed information for a list of video IDs"""
    if not video_ids:
        return {'items': []}
    
    results = {'items': []}
    
    batch_size = 50
    total_batches = (len(video_ids) + batch_size - 1) // batch_size
    
    progress_bar = tqdm(total=total_batches, desc="Fetching video details", unit="batch")
    
    for i in range(0, len(video_ids), batch_size):
        batch_ids = video_ids[i:i+batch_size]
        try:
            batch_details = execute_with_retry(youtube.videos().list(
                part='contentDetails,snippet,statistics',
                id=','.join(batch_ids)
            ))
            
            new_items = batch_details.get('items', [])
            results['items'].extend(new_items)
            
            progress_bar.update(1)
            progress_bar.set_postfix({"Videos": len(results['items'])})
            
            time.sleep(0.5)
        except Exception as e:
            progress_bar.write(f"Error fetching video details (batch {i//batch_size + 1}/{total_batches}): {e}")
    
    progress_bar.close()
    return results

def filter_videos(video_details: Dict[str, Any], min_duration_sec: int, max_duration_sec: int, 
                 min_views: Optional[int] = None, channels_blacklist: Optional[List[str]] = None, 
                 title_blacklist: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Filter videos based on duration, views, channel, and title criteria"""
    valid_videos = []
    
    channels_blacklist_lower = set([c.lower() for c in channels_blacklist]) if channels_blacklist else None
    title_blacklist_lower = [word.lower() for word in title_blacklist] if title_blacklist else None
    
    for item in video_details.get('items', []):
        try:
            channel_title = item['snippet']['channelTitle']
            title = item['snippet']['title']
            video_id = item['id']
            
            if channels_blacklist_lower and channel_title.lower() in channels_blacklist_lower:
                continue
                
            if title_blacklist_lower:
                title_lower = title.lower()
                if any(word in title_lower for word in title_blacklist_lower):
                    continue
                
            duration_iso = item['contentDetails']['duration']
            duration_sec = isodate.parse_duration(duration_iso).total_seconds()
            if not (min_duration_sec <= duration_sec <= max_duration_sec):
                continue
                
            view_count = int(item['statistics'].get('viewCount', '0'))
            if min_views is not None and view_count < min_views:
                continue
            
            likes = int(item['statistics'].get('likeCount', '0'))
            like_ratio = round((likes / view_count * 100), 2) if view_count > 0 else 0
                
            valid_videos.append({
                'title': title,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'duration': round(duration_sec / 60, 2),
                'duration_formatted': format_duration(duration_sec),
                'publishedAt': item['snippet']['publishedAt'],
                'published_formatted': format_date(item['snippet']['publishedAt']),
                'views': view_count,
                'views_formatted': humanize.intword(view_count),
                'likes': likes,
                'like_ratio': like_ratio,
                'channel': channel_title,
                'description': item['snippet'].get('description', '')[:100] + '...'
            })
            
        except Exception as e:
            print(f"Error processing video: {e}")
            continue
        
    return sorted(valid_videos, key=lambda x: int(x['views']), reverse=True)

# === FORMATTING FUNCTIONS ===
def format_duration(seconds: int) -> str:
    """Format seconds into HH:MM:SS or MM:SS"""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

def format_date(iso_date: str) -> str:
    """Format ISO date string to a readable format"""
    try:
        date_obj = datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%SZ")
        return date_obj.strftime("%b %d, %Y")
    except Exception:
        return iso_date

def truncate_text(text: str, max_length: int) -> str:
    """Safely truncate text to the specified length with ellipsis if needed"""
    if not text:
        return ""
    return text[:max_length] + ('...' if len(text) > max_length else '')

def format_output(videos: List[Dict[str, Any]], include_description: bool = False) -> str:
    """Format videos for display using tabulate"""
    if not videos:
        return "No videos found matching your criteria."
    
    try:
        terminal_width = os.get_terminal_size().columns
    except (AttributeError, OSError):
        terminal_width = 100
    
    title_max_length = min(50, max(20, terminal_width // 4))
    channel_max_length = min(20, max(10, terminal_width // 10))
    
    table_data = []
    for i, video in enumerate(videos, 1):
        row = [
            i, 
            truncate_text(video['title'], title_max_length),
            video['duration_formatted'],
            video['views_formatted'],
            f"{video['like_ratio']}%" if video['like_ratio'] > 0 else 'N/A',
            video['published_formatted'],
            truncate_text(video['channel'], channel_max_length),
            f"{video.get('composite_score', 'N/A')}",
            video['url']
        ]
        table_data.append(row)
    
    headers = ['#', 'Title', 'Duration', 'Views', 'Like Ratio', 'Published', 'Channel', 'Score', 'Link']
    if include_description:
        headers.append('Description')
        for video, row in zip(videos, table_data):
            row.append(video['description'])
    
    return tabulate(table_data, headers=headers, tablefmt="pretty")

def save_results(videos: List[Dict[str, Any]], query: str, output_format: str = 'json') -> None:
    """Save results to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"youtube_search_{query.replace(' ', '_')}_{timestamp}"
    
    try:
        if output_format == 'json':
            import json
            with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                json.dump(videos, f, indent=2, ensure_ascii=False)
                
        elif output_format == 'csv':
            import csv
            with open(f"{filename}.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'title', 'url', 'duration', 'duration_formatted', 
                    'views', 'views_formatted', 'likes', 'like_ratio',
                    'channel', 'publishedAt', 'published_formatted',
                    'gemini_score', 'gemini_analysis', 'composite_score',
                    'like_ratio_contribution', 'views_contribution'
                ])
                writer.writeheader()
                writer.writerows(videos)
                
        print(f"Results saved to {filename}.{output_format}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

# === COMMAND LINE INTERFACE ===
def create_args(query: str) -> argparse.Namespace:
    """Create argparse.Namespace with user query and force_refresh=True."""
    args = argparse.Namespace(
        query=query,
        days=DEFAULT_CONFIG['days_back'],
        min_duration=DEFAULT_CONFIG['min_duration_min'],
        max_duration=DEFAULT_CONFIG['max_duration_min'],
        min_views=None,
        exclude_channels=None,
        exclude_words=None,
        results=DEFAULT_CONFIG['max_results'],
        pages=DEFAULT_CONFIG['max_pages'],
        save=None,
        show_description=False,
        no_cache=False,
        force_refresh=True  # Hardcoded to ensure fresh results
    )
    return args

# === MAIN FUNCTION ===
def main():
    """Main function"""
    # Get query from user input (text or voice)
    query = get_user_query()
    args = create_args(query)

    quota_estimate = estimate_quota_usage(args.pages)
    print(f"Estimated quota usage: {quota_estimate} units") 
    
    if quota_estimate > 5000:
        print("⚠️ Warning: This operation may consume significant API quota")
        if input("Continue? (y/n): ").lower() != 'y':
            return

    api_key = load_api_key()
    youtube = build_youtube_client(api_key)
    
    published_after = (datetime.utcnow() - timedelta(days=args.days)).isoformat("T") + "Z"
    
    min_duration_sec = args.min_duration * 60
    max_duration_sec = args.max_duration * 60
    
    channels_blacklist = args.exclude_channels.split(',') if args.exclude_channels else None
    title_blacklist = args.exclude_words.split(',') if args.exclude_words else None
    
    cache_key = get_cache_key(
        args.query, 
        args.days, 
        args.min_duration, 
        args.max_duration,
        args.min_views,
        args.exclude_channels,
        args.exclude_words
    )
    
    results = None
    if not args.no_cache and not args.force_refresh:
        results = get_cached_results(cache_key, DEFAULT_CONFIG['cache_time'])
    
    if results is None:
        print(f"Searching for '{args.query}' videos from the past {args.days} days...")
        video_ids = search_videos(youtube, args.query, published_after, args.pages)
        
        if not video_ids:
            print("No videos found matching your query.")
            return
            
        print(f"Found {len(video_ids)} videos. Fetching details...")
        video_details = get_video_details(youtube, video_ids)
        
        results = filter_videos(
            video_details,
            min_duration_sec,
            max_duration_sec,
            args.min_views,
            channels_blacklist,
            title_blacklist
        )
        
        if not args.no_cache:
            cache_results(cache_key, results)
    
    if results:
        print("\nAnalyzing videos with Gemini 2.0 Flash...")
        analyzer = GeminiAnalyzer()
        analyzed_results = analyzer.analyze_videos(results[:args.results], args.query)
        best_video = analyzed_results[0] if analyzed_results else None
        
        if best_video and "composite_score" in best_video:
            print("\nBest Video (Based on Composite Score):")
            print(f"Title: {best_video['title']}")
            print(f"URL: {best_video['url']}")
            print(f"Composite Score: {best_video['composite_score']}/100")
            print(f" - Title Score: {best_video['gemini_score']}/100")
            print(f" - Like Ratio Contribution: {best_video['like_ratio_contribution']}/20")
            print(f" - Views Contribution: {best_video['views_contribution']}/20")
            print(f"Explanation: {best_video['gemini_analysis']}")
            print(f"Channel: {best_video['channel']}")
            print(f"Duration: {best_video['duration_formatted']}")
            print(f"Published: {best_video['published_formatted']}")
            print(f"Views: {best_video['views_formatted']}")
            print(f"Like Ratio: {best_video['like_ratio']}%")
        else:
            print("\nError selecting best video: No valid analysis returned")
            print("Check logs/gemini_responses.log for details.")
    
    if results:
        print(f"\nFound {len(results)} videos matching your criteria. Showing top {min(args.results, len(results))}:")
        print(format_output(analyzed_results[:args.results], args.show_description))
        
        if args.save:
            save_results(analyzed_results, args.query, args.save)
    else:
        print("No videos found matching your criteria.")

if __name__ == "__main__":
    try:
        clean_old_cache()
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        if isinstance(e, HttpError):
            print(f"Details: {e.content.decode()}")
        sys.exit(1)