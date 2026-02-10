"""
Provider health monitoring and status tracking.
Background task that periodically checks provider health.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import structlog

from src.providers.base import ProviderHealth, ProviderStatus, BaseProvider
from src.providers.factory import ProviderFactory

logger = structlog.get_logger()


class HealthMonitor:
    """Monitors health of all registered providers."""

    def __init__(
        self,
        check_interval: int = 30,
        unhealthy_threshold: int = 3,
        recovery_checks: int = 2,
    ):
        """
        Initialize health monitor.

        Args:
            check_interval: Seconds between health checks
            unhealthy_threshold: Consecutive failures before marking unhealthy
            recovery_checks: Consecutive successes before marking healthy
        """
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.recovery_checks = recovery_checks

        self._health_status: Dict[str, ProviderHealth] = {}
        self._failure_counts: Dict[str, int] = {}
        self._recovery_counts: Dict[str, int] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the health monitoring background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("health_monitor_started", interval=self.check_interval)

    async def stop(self):
        """Stop the health monitoring background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("health_monitor_stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_providers()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_monitor_error", error=str(e))
                await asyncio.sleep(5)

    async def _check_all_providers(self):
        """Check health of all registered providers."""
        providers = ProviderFactory.get_all_providers()

        tasks = []
        for name, provider in providers.items():
            tasks.append(self._check_provider(name, provider))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_provider(self, name: str, provider: BaseProvider):
        """Check health of a single provider."""
        try:
            health = await provider.health_check()

            if health.status == ProviderStatus.HEALTHY:
                self._failure_counts[name] = 0
                self._recovery_counts[name] = self._recovery_counts.get(name, 0) + 1

                # Check if recovered from unhealthy
                if self._recovery_counts[name] >= self.recovery_checks:
                    if name in self._health_status and \
                       self._health_status[name].status != ProviderStatus.HEALTHY:
                        logger.info(
                            "provider_recovered",
                            provider=name,
                            latency_ms=health.latency_ms
                        )
            else:
                self._recovery_counts[name] = 0
                self._failure_counts[name] = self._failure_counts.get(name, 0) + 1

                if self._failure_counts[name] >= self.unhealthy_threshold:
                    health.status = ProviderStatus.UNHEALTHY
                    logger.warning(
                        "provider_unhealthy",
                        provider=name,
                        failure_count=self._failure_counts[name],
                        last_error=health.last_error
                    )

            self._health_status[name] = health

        except Exception as e:
            logger.error("health_check_failed", provider=name, error=str(e))
            self._failure_counts[name] = self._failure_counts.get(name, 0) + 1

            if self._failure_counts[name] >= self.unhealthy_threshold:
                self._health_status[name] = ProviderHealth(
                    provider_name=name,
                    status=ProviderStatus.UNHEALTHY,
                    last_error=str(e)
                )

    def get_health(self, provider_name: str) -> Optional[ProviderHealth]:
        """Get current health status for a provider."""
        return self._health_status.get(provider_name)

    def get_all_health(self) -> Dict[str, ProviderHealth]:
        """Get health status for all providers."""
        return self._health_status.copy()

    def is_healthy(self, provider_name: str) -> bool:
        """Check if a provider is healthy."""
        health = self._health_status.get(provider_name)
        if not health:
            return True  # Assume healthy if not checked yet
        return health.status == ProviderStatus.HEALTHY

    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy provider names."""
        healthy = []
        for name, health in self._health_status.items():
            if health.status in (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED):
                healthy.append(name)

        # Include providers not yet checked
        all_providers = ProviderFactory.get_available_provider_names()
        for name in all_providers:
            if name not in self._health_status:
                healthy.append(name)

        return healthy

    def get_status_summary(self) -> Dict:
        """Get summary of all provider statuses."""
        providers = ProviderFactory.get_available_provider_names()
        summary = {
            "total_providers": len(providers),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0,
            "providers": {}
        }

        for name in providers:
            health = self._health_status.get(name)
            if health:
                status = health.status.value
                if health.status == ProviderStatus.HEALTHY:
                    summary["healthy"] += 1
                elif health.status == ProviderStatus.DEGRADED:
                    summary["degraded"] += 1
                else:
                    summary["unhealthy"] += 1

                summary["providers"][name] = {
                    "status": status,
                    "latency_ms": health.latency_ms,
                    "last_check": health.last_check.isoformat() if health.last_check else None,
                    "success_rate": health.success_rate,
                    "last_error": health.last_error
                }
            else:
                summary["unknown"] += 1
                summary["providers"][name] = {
                    "status": "unknown",
                    "latency_ms": None,
                    "last_check": None
                }

        return summary


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


async def start_health_monitoring():
    """Start global health monitoring."""
    monitor = get_health_monitor()
    await monitor.start()


async def stop_health_monitoring():
    """Stop global health monitoring."""
    monitor = get_health_monitor()
    await monitor.stop()
