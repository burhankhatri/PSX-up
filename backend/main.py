#!/usr/bin/env python3
"""
PSX Fortune Teller API - FastAPI Backend
AI-powered stock predictions for Pakistan Stock Exchange
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

# ============================================================================
# SETUP
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import Stock Analyzer Logic
try:
    from backend.stock_analyzer_fixed import (
        check_data as check_stock_data,
        analyze_stock as start_stock_analysis,
        websocket_progress as stock_websocket,
        StockRequest,
        progress_data
    )
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Stock Analyzer import error: {e}")
    ANALYZER_AVAILABLE = False

# ============================================================================
# APP
# ============================================================================
app = FastAPI(
    title="PSX Fortune Teller API",
    description="AI-powered stock predictions for Pakistan Stock Exchange",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Web Directory
WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
    print("Web directory mounted")

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Redirect to Fortune Teller UI"""
    return RedirectResponse(url="/analyzer")

@app.get("/analyzer")
async def analyzer_page():
    """Serve the Fortune Teller UI"""
    return FileResponse(WEB_DIR / "stock_analyzer.html")

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ============================================================================
# STOCK ANALYZER API
# ============================================================================

if ANALYZER_AVAILABLE:
    @app.post("/api/check-data")
    async def api_check_data(request: StockRequest):
        return await check_stock_data(request)

    @app.post("/api/analyze-stock")
    async def api_analyze_stock(request: StockRequest):
        return await start_stock_analysis(request)

    @app.websocket("/ws/progress/{job_id}")
    async def ws_progress(websocket: WebSocket, job_id: str):
        await stock_websocket(websocket, job_id)

    class BatchAnalyzeRequest(BaseModel):
        symbols: str  # Comma-separated symbols like "LUCK,EFERT,PPL"
        horizon: Union[int, str] = 21
        enable_geo_features: Optional[bool] = None

    @app.post("/api/batch-analyze")
    async def api_batch_analyze(request: BatchAnalyzeRequest):
        """Start batch analysis for multiple symbols. Returns independent job_ids."""
        symbols = list(dict.fromkeys(
            s.strip().upper() for s in request.symbols.split(",") if s.strip()
        ))
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")
        if len(symbols) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols per batch")

        jobs = []
        for symbol in symbols:
            stock_request = StockRequest(
                symbol=symbol,
                horizon=request.horizon,
                enable_geo_features=request.enable_geo_features,
            )
            result = await start_stock_analysis(stock_request)
            jobs.append({"symbol": symbol, "job_id": result["job_id"]})

        return {"jobs": jobs, "total": len(jobs)}

    print("Stock Analyzer routes registered")

# ============================================================================
# KSE100 ANALYZER
# ============================================================================

try:
    from backend.kse100_analyzer import analyze_kse100
    KSE100_AVAILABLE = True
    print("KSE100 analyzer available")
except ImportError as e:
    KSE100_AVAILABLE = False
    print(f"WARNING: KSE100 analyzer not available: {e}")

# ============================================================================
# UNIFIED ANALYZE ENDPOINT (Stocks + KSE100)
# ============================================================================

class AnalyzeRequest(BaseModel):
    symbol: str  # Stock symbol or 'KSE100'

@app.post("/api/analyze")
async def api_analyze(request: AnalyzeRequest):
    """
    Unified analyze endpoint that handles both stocks and KSE100.
    Returns results in the same format for UI compatibility.
    This is a synchronous endpoint (for simpler UI integration).
    For async progress updates, use /api/analyze-stock with WebSocket.
    """
    symbol = request.symbol.upper().strip()
    
    # Handle KSE100
    if symbol == 'KSE100':
        if not KSE100_AVAILABLE:
            raise HTTPException(status_code=503, detail="KSE100 analyzer not available")
        
        try:
            # Run KSE100 analysis (synchronous)
            result = analyze_kse100(horizon=365)
            
            # Format response to match stock analyzer format
            return {
                "symbol": "KSE100",
                "current_price": result.get('current_price', 0),
                "model": result.get('model', 'Research Model'),
                "model_performance": result.get('model_performance', {}),
                "daily_predictions": result.get('daily_predictions', []),
                "historical_data": result.get('historical_data', []),
                "generated_at": result.get('generated_at'),
                "wavelet_denoising": True
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"KSE100 analysis failed: {str(e)}")
    
    # Handle regular stocks - try to load from cache first, otherwise suggest WebSocket endpoint
    if not ANALYZER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Stock analyzer not available")
    
    # Check for cached predictions
    data_dir = BASE_DIR / "data"
    cached_files = [
        data_dir / f"{symbol}_research_predictions_2026.json",
        data_dir / f"{symbol}_sota_predictions_2026.json"
    ]
    
    for cached_file in cached_files:
        if cached_file.exists():
            try:
                with open(cached_file, 'r') as f:
                    data = json.load(f)
                
                # Load historical data
                hist_file = data_dir / f"{symbol}_historical_with_indicators.json"
                historical_data = []
                if hist_file.exists():
                    with open(hist_file, 'r') as hf:
                        hist_raw = json.load(hf)
                        historical_data = [{"Date": h['Date'], "Close": h['Close']} for h in hist_raw[-180:]]
                
                # Get current price
                current_price = 0
                if historical_data:
                    current_price = historical_data[-1].get('Close', 0)
                elif data.get('daily_predictions'):
                    current_price = data['daily_predictions'][0].get('predicted_price', 0)
                
                return {
                    "symbol": symbol,
                    "current_price": current_price,
                    "model": data.get('model', 'Research Model'),
                    "model_performance": data.get('metrics', data.get('model_performance', {})),
                    "daily_predictions": data.get('daily_predictions', []),
                    "historical_data": historical_data,
                    "generated_at": data.get('generated_at'),
                    "wavelet_denoising": data.get('wavelet_denoising', True)
                }
            except Exception as e:
                continue
    
    # No cache found - suggest using WebSocket endpoint
    raise HTTPException(
        status_code=404, 
        detail=f"No cached analysis for {symbol}. Please use /api/analyze-stock endpoint with WebSocket for real-time analysis."
    )

# ============================================================================
# SCREENER & SENTIMENT
# ============================================================================

try:
    from backend.stock_screener import run_screener, screen_stock, PSX_MAJOR_STOCKS
    from backend.sentiment_analyzer import get_stock_sentiment
    from backend.hot_stocks import get_cached_trending_stocks, get_hot_stocks_for_homepage
    
    # Cache for screener results
    SCREENER_CACHE_FILE = BASE_DIR / "data" / "screener_cache.json"
    
    def load_screener_cache():
        """Load cached screener results"""
        if SCREENER_CACHE_FILE.exists():
            try:
                with open(SCREENER_CACHE_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def save_screener_cache(results):
        """Save screener results to cache"""
        cache_data = {
            "cached_at": datetime.now().isoformat(),
            "top_picks": results
        }
        try:
            with open(SCREENER_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"WARNING: Could not save screener cache: {e}")

    @app.get("/api/screener")
    async def screener(limit: int = 10, refresh: bool = False):
        """Get top stock picks from screener. Uses cache unless refresh=true."""
        try:
            # Return cached results if available and not refreshing
            if not refresh:
                cached = load_screener_cache()
                if cached and cached.get('top_picks'):
                    return {
                        "success": True, 
                        "top_picks": cached['top_picks'][:limit],
                        "cached_at": cached.get('cached_at'),
                        "from_cache": True
                    }
            
            # Run fresh scan
            results = run_screener(limit=min(limit, 50))
            
            # Save to cache
            save_screener_cache(results)
            
            return {"success": True, "top_picks": results, "from_cache": False}
        except Exception as e:
            return {"success": False, "error": str(e), "top_picks": []}

    @app.get("/api/trending")
    async def trending():
        """Get trending/hot stocks"""
        try:
            stocks = get_cached_trending_stocks()
            return {"success": True, "stocks": stocks}
        except Exception as e:
            return {"success": False, "error": str(e), "stocks": []}

    @app.get("/api/sentiment/{symbol}")
    async def sentiment(symbol: str):
        """Get AI sentiment analysis for a stock"""
        try:
            result = get_stock_sentiment(symbol.upper())
            return {"success": True, "sentiment": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    print("Screener and sentiment routes registered")
except ImportError as e:
    print(f"WARNING: Screener/Sentiment not available: {e}")

# ============================================================================
# COMMODITY PREDICTION (Silver & Gold)
# ============================================================================

try:
    from backend.commodity_predictor import (
        analyze_commodity,
        get_commodity_quick_data,
        load_commodity_analysis,
        COMMODITY_CONFIG
    )
    COMMODITY_AVAILABLE = True
    print("Commodity predictor available")
except ImportError as e:
    COMMODITY_AVAILABLE = False
    print(f"WARNING: Commodity predictor not available: {e}")

if COMMODITY_AVAILABLE:
    class CommodityRequest(BaseModel):
        commodity: str  # 'silver' or 'gold'
    
    commodity_jobs = {}
    
    # NOTE: Static routes must come BEFORE dynamic {symbol} route
    @app.get("/api/commodity/list")
    async def list_commodities():
        """List available commodities for prediction"""
        return {
            "success": True,
            "commodities": [
                {
                    "id": key,
                    "name": cfg['name'],
                    "emoji": cfg['emoji'],
                    "color": cfg['color']
                }
                for key, cfg in COMMODITY_CONFIG.items()
            ]
        }
    
    @app.get("/api/commodity/{symbol}")
    async def get_commodity(symbol: str):
        """Get quick price data for a commodity (silver or gold)"""
        result = get_commodity_quick_data(symbol.lower())
        if 'error' in result:
            return {"success": False, "error": result['error']}
        return {"success": True, **result}
    
    @app.post("/api/commodity/analyze")
    async def start_commodity_analysis(request: CommodityRequest):
        """Start commodity analysis (returns job_id for WebSocket)"""
        commodity = request.commodity.lower()
        if commodity not in COMMODITY_CONFIG:
            raise HTTPException(status_code=400, detail=f"Unknown commodity: {commodity}")
        
        job_id = f"{commodity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        commodity_jobs[job_id] = {
            'status': 'starting',
            'commodity': commodity
        }
        return {"job_id": job_id, "commodity": commodity}
    
    @app.websocket("/ws/commodity/{job_id}")
    async def ws_commodity_progress(websocket: WebSocket, job_id: str):
        """WebSocket for commodity analysis progress"""
        await websocket.accept()
        
        try:
            if job_id not in commodity_jobs:
                await websocket.send_json({
                    'stage': 'error',
                    'progress': 0,
                    'message': f'Job ID {job_id} not found'
                })
                await websocket.close()
                return
            
            commodity = commodity_jobs[job_id]['commodity']
            
            await websocket.send_json({
                'stage': 'starting',
                'progress': 5,
                'message': f'Starting {commodity.title()} analysis...',
            })
            
            # Check for cached analysis first
            cached = load_commodity_analysis(commodity)
            if cached:
                await websocket.send_json({
                    'stage': 'complete',
                    'progress': 100,
                    'message': 'Loaded cached analysis',
                    'results': cached
                })
                await websocket.close()
                return
            
            await websocket.send_json({
                'stage': 'fetching',
                'progress': 15,
                'message': f'Fetching {commodity.title()} price data...',
            })
            
            await websocket.send_json({
                'stage': 'indicators',
                'progress': 35,
                'message': '🔬 Fetching AI/GPU demand indicators...'
            })
            
            await websocket.send_json({
                'stage': 'macro',
                'progress': 50,
                'message': '💵 Fetching macro indicators (USD, Fed rates)...'
            })
            
            await websocket.send_json({
                'stage': 'training',
                'progress': 70,
                'message': f'Training {commodity.title()} prediction model...',
            })
            
            # Run full analysis
            result = analyze_commodity(commodity)
            
            await websocket.send_json({
                'stage': 'complete',
                'progress': 100,
                'message': 'Analysis complete.',
                'results': result
            })
            
        except Exception as e:
            await websocket.send_json({
                'stage': 'error',
                'progress': 0,
                'message': f'Error: {str(e)}'
            })
        finally:
            await websocket.close()
    
    @app.get("/api/commodity/history/{commodity}")
    async def get_commodity_history(commodity: str):
        """Get cached commodity analysis if available"""
        cached = load_commodity_analysis(commodity.lower())
        if cached:
            return {"success": True, "analysis": cached}
        return {"success": False, "error": "No cached analysis available"}

# ============================================================================
# HISTORY PERSISTENCE
# ============================================================================

@app.get("/api/history")
async def get_prediction_history():
    """Get list of saved predictions with rich metadata for dashboard display."""
    try:
        data_dir = BASE_DIR / "data"

        sota_files = list(data_dir.glob("*_sota_predictions_2026.json"))
        research_files = list(data_dir.glob("*_research_predictions_2026.json"))
        all_files = sota_files + research_files
        all_files = sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)

        history = []
        seen_symbols = set()

        for f in all_files:
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    symbol = data.get('symbol')

                    if symbol in seen_symbols:
                        continue
                    seen_symbols.add(symbol)

                    daily_preds = data.get('daily_predictions', [])
                    last_pred = daily_preds[-1] if daily_preds else {}
                    first_pred = daily_preds[0] if daily_preds else {}

                    entry = {
                        "filename": f.name,
                        "symbol": symbol,
                        "generated_at": data.get('generated_at'),
                        "current_price": first_pred.get('predicted_price', 0),
                        "predicted_return": last_pred.get('upside_potential', 0),
                        "model": data.get('model', 'Unknown'),
                        "prediction_days": len(daily_preds),
                    }

                    # Model performance
                    metrics = data.get('metrics', {})
                    entry["model_performance"] = {
                        "r2": metrics.get('r2'),
                        "trend_accuracy": metrics.get('trend_accuracy', metrics.get('ensemble_accuracy')),
                        "mape": metrics.get('mape'),
                    }

                    # Prediction reasoning
                    reasoning = data.get('prediction_reasoning', {})
                    entry["direction"] = reasoning.get('direction', 'NEUTRAL')
                    entry["direction_emoji"] = reasoning.get('emoji', '')
                    entry["bullish_count"] = reasoning.get('bullish_count', 0)
                    entry["bearish_count"] = reasoning.get('bearish_count', 0)

                    # Complete analysis data (sentiment, forecast summary)
                    complete_file = data_dir / f"{symbol}_complete_analysis.json"
                    if complete_file.exists():
                        try:
                            with open(complete_file, 'r') as cf:
                                complete = json.load(cf)
                            sentiment = complete.get('sentiment', {})
                            entry["sentiment_signal"] = sentiment.get('signal', 'NEUTRAL')
                            entry["sentiment_emoji"] = sentiment.get('signal_emoji', '')
                            forecast_summary = complete.get('forecast_summary', {})
                            entry["overall_direction"] = forecast_summary.get('overall_direction', 'UNKNOWN')
                            entry["bullish_months"] = forecast_summary.get('bullish_months', 0)
                            entry["bearish_months"] = forecast_summary.get('bearish_months', 0)
                        except Exception:
                            pass

                    history.append(entry)
            except Exception:
                continue

        # Prediction log entries for the dashboard table
        prediction_log = []
        try:
            from backend.prediction_logger import get_prediction_logger
            logger = get_prediction_logger()
            prediction_log = logger.get_recent_predictions(limit=50)
        except Exception:
            pass

        return {"success": True, "history": history, "prediction_log": prediction_log}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/history/{filename}")
async def get_prediction_file(filename: str):
    """Get saved prediction with historical data for charting.
    
    Returns complete data structure matching WebSocket response format.
    """
    # Allow both SOTA and Research model prediction files
    valid_suffixes = ("_sota_predictions_2026.json", "_research_predictions_2026.json")
    if not filename.endswith(valid_suffixes) or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    file_path = BASE_DIR / "data" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Load historical data for charting
    symbol = data.get('symbol', '')
    hist_file = BASE_DIR / "data" / f"{symbol}_historical_with_indicators.json"
    historical_data = []
    
    if hist_file.exists():
        with open(hist_file, 'r') as hf:
            hist_raw = json.load(hf)
            historical_data = [{"Date": h['Date'], "Close": h['Close']} for h in hist_raw[-180:]]
    
    daily_preds = data.get('daily_predictions', [])
    first_pred = daily_preds[0] if daily_preds else {}
    
    # Get current price from historical data or first prediction
    current_price = 0
    if historical_data:
        current_price = historical_data[-1].get('Close', 0)
    if not current_price and first_pred:
        current_price = first_pred.get('predicted_price', 0)
    
    # Load cached sentiment if available
    sentiment = {}
    news_cache_file = BASE_DIR / "data" / "news_cache" / f"{symbol}_news.json"
    if news_cache_file.exists():
        try:
            with open(news_cache_file, 'r') as sf:
                sentiment_data = json.load(sf)
                sentiment = {
                    'signal': sentiment_data.get('signal', 'NEUTRAL'),
                    'signal_emoji': sentiment_data.get('signal_emoji', '🟡'),
                    'summary': sentiment_data.get('summary', 'No sentiment analysis available.'),
                    'sentiment_score': sentiment_data.get('sentiment_score', 0),
                    'confidence': sentiment_data.get('confidence', 0),
                    'recent_news': [
                        {
                            'title': n.get('title', ''),
                            'source': n.get('source_name', n.get('source', 'News')),
                            'published_at': n.get('date', 'Recently')
                        }
                        for n in sentiment_data.get('news_items', [])[:5]
                    ]
                }
        except Exception as e:
            print(f"Error loading sentiment cache: {e}")

    # Load monthly_forecast and forecast_summary from complete analysis if available
    complete_file = BASE_DIR / "data" / f"{symbol}_complete_analysis.json"
    monthly_forecast = []
    forecast_summary = {}
    daily_predictions_without_geo = daily_preds
    daily_predictions_with_geo = []
    geo_comparison = {
        "enabled": False,
        "applied": False,
        "labels": {
            "baseline": "Without Geo Features",
            "geo": "With Geo Features",
        },
    }
    if complete_file.exists():
        try:
            with open(complete_file, 'r') as cf:
                complete_data = json.load(cf)
            monthly_forecast = complete_data.get('monthly_forecast', [])
            forecast_summary = complete_data.get('forecast_summary', {})
            daily_predictions_without_geo = complete_data.get('daily_predictions_without_geo', daily_preds)
            daily_predictions_with_geo = complete_data.get('daily_predictions_with_geo', [])
            geo_comparison = complete_data.get('geo_comparison', geo_comparison)
        except Exception:
            pass

    # Return complete structure matching WebSocket response
    return {
        "symbol": symbol,
        "current_price": current_price,
        "model": data.get('model', 'SOTA Ensemble'),
        "model_performance": data.get('metrics', {}),
        "daily_predictions": daily_preds,
        "daily_predictions_without_geo": daily_predictions_without_geo,
        "daily_predictions_with_geo": daily_predictions_with_geo,
        "geo_comparison": geo_comparison,
        "historical_data": historical_data,
        "sentiment": sentiment,
        "prediction_reasoning": data.get('prediction_reasoning'),  # Include reasoning if available
        "monthly_forecast": monthly_forecast,
        "forecast_summary": forecast_summary,
    }


@app.get("/api/prediction-tuning-report")
async def get_prediction_tuning_report(refresh: bool = False):
    """
    Return latest prediction tuning A/B report and drift snapshot.
    - refresh=false: read cached report if present, generate only if missing
    - refresh=true: regenerate report before returning
    """
    try:
        report_path = BASE_DIR / "data" / "prediction_logs" / "prediction_tuning_report.json"
        log_path = BASE_DIR / "data" / "prediction_logs" / "prediction_log.json"

        if not log_path.exists():
            return {
                "success": False,
                "error": "Prediction log not found",
                "report": None,
                "drift": None,
            }

        from backend.prediction_tuning import write_ab_report, drift_snapshot

        if refresh or not report_path.exists():
            write_ab_report(str(log_path), str(report_path))

        with open(report_path, "r") as rf:
            report = json.load(rf)

        drift = None
        try:
            drift = drift_snapshot(str(log_path))
        except Exception as e:
            drift = {"error": str(e)}

        return {
            "success": True,
            "report": report,
            "drift": drift,
            "report_path": str(report_path),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "report": None,
            "drift": None,
        }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("� PSX FORTUNE TELLER API")
    print("=" * 60)
    print("UI: http://localhost:8000/analyzer")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
