import os
import re
import time
import json
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from datetime import datetime

class GeminiAnalyzer:
    def __init__(self):
        self.configured = False
        self.model = None
        self.last_call_time = None
        self.RATE_LIMIT_DELAY = 1.2  # seconds between calls
        self.WEIGHTS = {
            'gemini_score': 0.6,
            'like_ratio': 0.2,
            'views': 0.2
        }
        self.MAX_RETRIES = 3

    def configure(self) -> None:
        """Configure Gemini API with safety settings."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash",
            safety_settings={
                "HARASSMENT": "block_none",
                "HATE_SPEECH": "block_none",
                "SEXUAL": "block_none",
                "DANGEROUS": "block_none"
            }
        )
        self.configured = True

    def _enforce_rate_limit(self) -> None:
        """Prevent API rate limit issues."""
        if self.last_call_time:
            elapsed = (datetime.now() - self.last_call_time).total_seconds()
            if elapsed < self.RATE_LIMIT_DELAY:
                time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_call_time = datetime.now()

    def _log_response(self, text: str, error: str = None) -> None:
        """Log the raw Gemini response for debugging."""
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "gemini_responses.log"), "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now()}] {'ERROR: ' + error if error else 'Response'}:\n{text[:2000]}\n{'='*50}\n")

    def analyze_videos(self, videos: List[Dict], query: str) -> List[Dict]:
        """Analyze YouTube video titles, like ratio, and views using Gemini."""
        if not self.configured:
            self.configure()

        if not videos:
            return []

        # Simplify prompt to reduce complexity
        prompt = f"""Analyze these YouTube videos for the search query "{query}".
Evaluate each video based on:
1. Title relevance to the query.
2. Title clarity and appeal.
3. Like ratio (higher is better).
4. Views (higher is better).
Assign a score (0-100, integer) reflecting title quality and relevance, adjusted by engagement (like ratio) and popularity (views).

Output a valid JSON array:
{{
  "analysis": [
    {{
      "index": 0,
      "score": 85,
      "rationale": "Relevant title with high engagement."
    }}
  ]
}}

Example for query "learn python":
{{
  "analysis": [
    {{
      "index": 0,
      "score": 92,
      "rationale": "Beginner-focused title, high like ratio, many views."
    }},
    {{
      "index": 1,
      "score": 75,
      "rationale": "Advanced topic, moderate engagement."
    }}
  ]
}}

Instructions:
- Output ONLY valid JSON (no markdown, no extra text).
- Ensure one entry per video.
- Keep rationales concise (1-2 sentences).
- Handle special characters in titles safely.

Videos:
{chr(10).join(f"{i}. {v['title'][:100]} (Like Ratio: {v['like_ratio']}%, Views: {v['views_formatted']})" for i, v in enumerate(videos))}"""

        for attempt in range(self.MAX_RETRIES):
            try:
                self._enforce_rate_limit()
                response = self.model.generate_content(prompt)
                self._log_response(response.text)  # Log successful response
                return self._process_response(videos, response.text)
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.MAX_RETRIES}: {str(e)}"
                self._log_response(f"Error: {str(e)}", error=error_msg)
                print(f"⚠️ Gemini analysis failed: {error_msg}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print("⚠️ Max retries reached. Using fallback.")
                    return self._create_fallback(videos)

    def _process_response(self, videos: List[Dict], text: str) -> List[Dict]:
        """Parse response with multiple fallback strategies."""
        # Strategy 1: Direct JSON parse
        try:
            data = json.loads(text.strip())
            return self._format_results(videos, data)
        except json.JSONDecodeError as e:
            self._log_response(text, f"Direct JSON parse failed: {str(e)}")
            pass

        # Strategy 2: Extract JSON from markdown
        try:
            json_str = re.search(r'```json\n(.*?)\n```', text, re.DOTALL).group(1)
            data = json.loads(json_str)
            return self._format_results(videos, data)
        except (AttributeError, json.JSONDecodeError) as e:
            self._log_response(text, f"Markdown JSON parse failed: {str(e)}")
            pass

        # Strategy 3: Extract JSON-like structure with fixes
        try:
            text = text.replace("'", '"')  # Fix single quotes
            text = re.sub(r'//.*?\n', '', text)  # Remove comments
            text = re.sub(r',\s*}', '}', text)  # Fix trailing commas
            text = re.sub(r',\s*]', ']', text)  # Fix trailing commas in arrays
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != -1:
                data = json.loads(text[start:end])
                return self._format_results(videos, data)
        except json.JSONDecodeError as e:
            self._log_response(text, f"JSON-like parse failed: {str(e)}")
            pass

        # Strategy 4: Partial JSON recovery
        try:
            # Try to extract valid analysis entries
            matches = re.findall(r'\{[^}]*"index"\s*:\s*\d+[^}]*\}', text)
            analysis = []
            for match in matches:
                try:
                    analysis.append(json.loads(match))
                except json.JSONDecodeError:
                    continue
            if analysis:
                data = {"analysis": analysis}
                return self._format_results(videos, data)
        except Exception as e:
            self._log_response(text, f"Partial JSON recovery failed: {str(e)}")
            pass

        print("⚠️ All parsing strategies failed")
        return self._create_fallback(videos)

    def _format_results(self, videos: List[Dict], data: Dict) -> List[Dict]:
        """Format Gemini analysis results and compute composite score."""
        if "analysis" not in data or not isinstance(data["analysis"], list):
            print("⚠️ Invalid response format: missing or invalid 'analysis' field")
            self._log_response(json.dumps(data), "Invalid analysis format")
            return self._create_fallback(videos)

        max_views = max(v['views'] for v in videos) if videos else 1
        results = []
        for item in data["analysis"]:
            try:
                index = item.get("index")
                if not isinstance(index, int) or index >= len(videos):
                    raise ValueError(f"Invalid index: {index}")
                video = videos[index]
                gemini_score = self._normalize_score(item.get("score"))
                like_ratio = video.get('like_ratio', 0)
                views = video.get('views', 0)
                normalized_views = (views / max_views * 100) if max_views > 0 else 0
                composite_score = (
                    self.WEIGHTS['gemini_score'] * gemini_score +
                    self.WEIGHTS['like_ratio'] * like_ratio +
                    self.WEIGHTS['views'] * normalized_views
                )
                results.append({
                    **video,
                    "gemini_score": gemini_score,
                    "gemini_analysis": item.get("rationale", "No rationale provided"),
                    "composite_score": round(composite_score, 2),
                    "like_ratio_contribution": round(self.WEIGHTS['like_ratio'] * like_ratio, 2),
                    "views_contribution": round(self.WEIGHTS['views'] * normalized_views, 2)
                })
            except Exception as e:
                print(f"⚠️ Item parsing failed: {str(e)}")
                continue

        return sorted(results, key=lambda x: x["composite_score"], reverse=True) or self._create_fallback(videos)

    def _create_fallback(self, videos: List[Dict]) -> List[Dict]:
        """Generate neutral results when analysis fails."""
        max_views = max(v['views'] for v in videos) if videos else 1
        return [{
            **v,
            "gemini_score": 50,
            "gemini_analysis": "Could not analyze content",
            "composite_score": round(
                self.WEIGHTS['gemini_score'] * 50 +
                self.WEIGHTS['like_ratio'] * v.get('like_ratio', 0) +
                self.WEIGHTS['views'] * (v.get('views', 0) / max_views * 100 if max_views > 0 else 0),
                2
            ),
            "like_ratio_contribution": round(self.WEIGHTS['like_ratio'] * v.get('like_ratio', 0), 2),
            "views_contribution": round(self.WEIGHTS['views'] * (v.get('views', 0) / max_views * 100 if max_views > 0 else 0), 2)
        } for v in videos]

    def _normalize_score(self, raw_score: Any) -> float:
        """Ensures score is always 0-100."""
        try:
            return max(0, min(100, float(raw_score)))
        except:
            return 50.0

    def get_top_video(self, videos: List[Dict], query: str) -> Optional[Dict]:
        """Convenience method to get the single best video based on composite score."""
        analyzed = self.analyze_videos(videos[:50], query)
        return analyzed[0] if analyzed else None

if __name__ == "__main__":
    analyzer = GeminiAnalyzer()
    test_videos = [
        {"title": "Python Tutorial for Beginners 2025", "url": "http://example.com/1", "like_ratio": 95.5, "views": 1200000, "views_formatted": "1.2M"},
        {"title": "Advanced Python Tricks", "url": "http://example.com/2", "like_ratio": 90.2, "views": 800000, "views_formatted": "800K"}
    ]
    
    results = analyzer.analyze_videos(test_videos, "learn python")
    print(json.dumps(results, indent=2))