import * as echarts from "echarts/core"
import {BarChart, LineChart} from "echarts/charts"
import {GridComponent, TooltipComponent} from "echarts/components"
import {CanvasRenderer} from "echarts/renderers"
import type {EChartsOption} from "echarts"
import {useEffect, useRef} from "react"

echarts.use([LineChart, BarChart, GridComponent, TooltipComponent, CanvasRenderer])


export function EChart({option, ariaLabel}: {option: EChartsOption; ariaLabel: string}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current)
    chart.setOption(option)
    const resize = () => chart.resize()
    window.addEventListener("resize", resize)
    return () => { window.removeEventListener("resize", resize); chart.dispose() }
  }, [option])
  return <div className="chart" ref={ref} role="img" aria-label={ariaLabel}/>
}
