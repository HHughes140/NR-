import numpy as np
import pandas as pd


class DrawdownAnalyzer:
    """Drawdown computation and analysis."""

    @staticmethod
    def drawdown_series(returns: pd.Series) -> pd.DataFrame:
        """Compute running drawdown from peak."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return pd.DataFrame({
            "cumulative_return": cumulative,
            "running_max": running_max,
            "drawdown": drawdown,
        })

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Maximum peak-to-trough percentage decline (returned as negative)."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())

    @staticmethod
    def max_drawdown_duration(returns: pd.Series) -> int:
        """Longest drawdown period in trading days."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        is_drawdown = cumulative < running_max

        max_duration = 0
        current_duration = 0
        for in_dd in is_drawdown:
            if in_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        return max_duration

    @staticmethod
    def top_drawdowns(returns: pd.Series, n: int = 5) -> pd.DataFrame:
        """Find the N worst drawdown episodes."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        episodes = []
        in_dd = False
        start = None

        for i, (date, dd) in enumerate(drawdown.items()):
            if dd < 0 and not in_dd:
                in_dd = True
                start = date
            elif dd == 0 and in_dd:
                in_dd = False
                dd_slice = drawdown.loc[start:date]
                trough_date = dd_slice.idxmin()
                depth = float(dd_slice.min())
                duration = len(dd_slice)
                episodes.append({
                    "start_date": start,
                    "trough_date": trough_date,
                    "recovery_date": date,
                    "depth": depth,
                    "duration_days": duration,
                })

        # Handle ongoing drawdown
        if in_dd and start is not None:
            dd_slice = drawdown.loc[start:]
            trough_date = dd_slice.idxmin()
            depth = float(dd_slice.min())
            episodes.append({
                "start_date": start,
                "trough_date": trough_date,
                "recovery_date": None,
                "depth": depth,
                "duration_days": len(dd_slice),
            })

        df = pd.DataFrame(episodes)
        if df.empty:
            return df
        return df.nsmallest(n, "depth").reset_index(drop=True)

    @staticmethod
    def underwater_series(returns: pd.Series) -> pd.Series:
        """Drawdown percentage below high-water mark, for underwater chart."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max
