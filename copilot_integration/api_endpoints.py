"""
FastAPI endpoints for GitHub Copilot integration.

This module provides REST API endpoints that GitHub Copilot can call
to get strategic context and optimization suggestions.
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .copilot_api import CopilotStrategyAPI
from .strategy_database import get_strategy_database, StrategyResult


def create_copilot_api() -> FastAPI:
    """Create FastAPI application for Copilot integration."""
    
    app = FastAPI(
        title="Copilot Strategy Context API",
        description="API for GitHub Copilot to access historical strategy performance data",
        version="2.0.0"
    )
    
    # Add CORS middleware to allow requests from development tools
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Copilot Strategy Context API",
            "version": "2.0.0",
            "description": "Provides historical strategy context for GitHub Copilot",
            "endpoints": [
                "/api/strategy-context/{strategy_type}",
                "/api/validate-strategy",
                "/api/suggest-improvements/{strategy_id}",
                "/api/parameters/{strategy_type}",
                "/api/code-patterns/{strategy_type}"
            ]
        }
    
    @app.get("/api/strategy-context/{strategy_type}")
    async def get_strategy_context(
        strategy_type: str,
        include_code_patterns: bool = True,
        market_regime: Optional[str] = None
    ):
        """
        API endpoint que Copilot peut appeler pour obtenir le contexte.
        
        Args:
            strategy_type: Type of strategy ('moving_average', 'rsi', etc.)
            include_code_patterns: Whether to include code patterns
            market_regime: Optional market regime filter
        """
        try:
            context = CopilotStrategyAPI.get_strategy_insights(strategy_type)
            optimal_parameters = CopilotStrategyAPI.suggest_parameters(strategy_type, market_regime)
            
            response = {
                "strategy_type": strategy_type,
                "context": context,
                "optimal_parameters": optimal_parameters,
                "market_regime": market_regime,
                "timestamp": datetime.now().isoformat()
            }
            
            if include_code_patterns:
                response["code_patterns"] = CopilotStrategyAPI.get_code_patterns(strategy_type)
                response["implementation_hints"] = CopilotStrategyAPI.get_implementation_hints(strategy_type)
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")
    
    @app.get("/api/validate-strategy")
    async def validate_strategy(
        strategy_name: str = Query(..., description="Strategy name"),
        parameters: str = Query(..., description="Strategy parameters as JSON string")
    ):
        """Valide si une stratégie similaire existe déjà."""
        try:
            # Parse parameters
            params = json.loads(parameters)
            
            # Generate signature
            signature = _hash_parameters(strategy_name, params)
            
            # Check if strategy exists
            validation_result = CopilotStrategyAPI.check_strategy_exists(signature)
            
            return {
                "strategy_name": strategy_name,
                "parameters": params,
                "signature": signature,
                **validation_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in parameters")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")
    
    @app.get("/api/suggest-improvements/{strategy_id}")
    async def suggest_improvements(strategy_id: str):
        """Suggère des améliorations pour une stratégie existante."""
        try:
            db = get_strategy_database()
            
            # Find strategy by ID (simplified - in real implementation would be more robust)
            matching_results = [r for r in db.results if r.signature == strategy_id]
            
            if not matching_results:
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            strategy_result = matching_results[0]
            strategy_type = strategy_result.strategy_type
            
            # Get suggestions based on strategy type
            insights = db.get_strategy_performance_summary(strategy_type)
            implementation_hints = CopilotStrategyAPI.get_implementation_hints(strategy_type)
            
            return {
                "strategy_id": strategy_id,
                "current_performance": strategy_result.performance_metrics,
                "suggestions": insights.suggestions,
                "implementation_hints": implementation_hints,
                "optimal_parameters": insights.best_params,
                "timestamp": datetime.now().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating suggestions: {str(e)}")
    
    @app.get("/api/parameters/{strategy_type}")
    async def get_optimal_parameters(
        strategy_type: str,
        market_regime: Optional[str] = None
    ):
        """Get optimal parameters for a strategy type."""
        try:
            parameters = CopilotStrategyAPI.suggest_parameters(strategy_type, market_regime)
            
            return {
                "strategy_type": strategy_type,
                "market_regime": market_regime,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving parameters: {str(e)}")
    
    @app.get("/api/code-patterns/{strategy_type}")
    async def get_code_patterns(strategy_type: str):
        """Get successful code patterns for a strategy type."""
        try:
            patterns = CopilotStrategyAPI.get_code_patterns(strategy_type)
            implementation_hints = CopilotStrategyAPI.get_implementation_hints(strategy_type)
            
            return {
                "strategy_type": strategy_type,
                "code_patterns": patterns,
                "implementation_hints": implementation_hints,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving patterns: {str(e)}")
    
    @app.post("/api/add-result")
    async def add_strategy_result(result_data: Dict[str, Any]):
        """Add a new strategy result to the database."""
        try:
            # Validate required fields
            required_fields = ['strategy_type', 'parameters', 'performance_metrics']
            for field in required_fields:
                if field not in result_data:
                    raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
            
            # Create strategy result
            strategy_result = StrategyResult(
                strategy_type=result_data['strategy_type'],
                parameters=result_data['parameters'],
                performance_metrics=result_data['performance_metrics'],
                market_regime=result_data.get('market_regime'),
                timestamp=result_data.get('timestamp')
            )
            
            # Add to database
            db = get_strategy_database()
            db.add_result(strategy_result)
            
            return {
                "message": "Strategy result added successfully",
                "signature": strategy_result.signature,
                "timestamp": strategy_result.timestamp
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error adding result: {str(e)}")
    
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        db = get_strategy_database()
        
        return {
            "status": "healthy",
            "database_results": len(db.results),
            "timestamp": datetime.now().isoformat()
        }
    
    return app


def _hash_parameters(strategy_name: str, params: Dict[str, Any]) -> str:
    """Generate hash signature for strategy parameters."""
    import hashlib
    
    param_str = json.dumps(params, sort_keys=True)
    content = f"{strategy_name}_{param_str}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


# Create the FastAPI app instance
app = create_copilot_api()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)