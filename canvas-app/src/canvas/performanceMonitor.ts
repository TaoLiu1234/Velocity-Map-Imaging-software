// 性能监控工具
export class PerformanceMonitor {
  private static instance: PerformanceMonitor
  private measurements = new Map<string, number[]>()
  private enabled = false

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor()
    }
    return PerformanceMonitor.instance
  }

  enable() {
    this.enabled = true
  }

  disable() {
    this.enabled = false
  }

  startMeasurement(name: string) {
    if (!this.enabled) return
    this.measurements.set(name, [performance.now()])
  }

  endMeasurement(name: string) {
    if (!this.enabled) return
    const times = this.measurements.get(name)
    if (times && times.length === 1) {
      const duration = performance.now() - times[0]
      times.push(duration)
      console.log(`[Performance] ${name}: ${duration.toFixed(2)}ms`)
    }
  }

  getAverageTime(name: string): number {
    const times = this.measurements.get(name)
    if (!times || times.length <= 1) return 0
    const durations = times.slice(1)
    return durations.reduce((sum, time) => sum + time, 0) / durations.length
  }

  getStats() {
    const stats: Record<string, { avg: number; count: number; max: number }> = {}
    for (const [name, times] of this.measurements) {
      if (times.length > 1) {
        const durations = times.slice(1)
        stats[name] = {
          avg: durations.reduce((sum, time) => sum + time, 0) / durations.length,
          count: durations.length,
          max: Math.max(...durations)
        }
      }
    }
    return stats
  }

  clear() {
    this.measurements.clear()
  }
}

// React Hook for component performance monitoring
export function usePerformanceMonitor(componentName: string, enabled = true) {
  const monitor = PerformanceMonitor.getInstance()

  return {
    startRender: () => enabled && monitor.startMeasurement(`${componentName}-render`),
    endRender: () => enabled && monitor.endMeasurement(`${componentName}-render`),
    startUpdate: () => enabled && monitor.startMeasurement(`${componentName}-update`),
    endUpdate: () => enabled && monitor.endMeasurement(`${componentName}-update`),
  }
}

// Global performance monitor instance
export const performanceMonitor = PerformanceMonitor.getInstance()
