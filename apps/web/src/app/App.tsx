import {BrowserRouter, Route, Routes} from "react-router-dom"

import {AppShell} from "../layout/AppShell"
import {AlertsPage} from "../pages/AlertsPage"
import {DashboardPage} from "../pages/DashboardPage"
import {DefectsPage} from "../pages/DefectsPage"
import {InspectionPage} from "../pages/InspectionPage"
import {ModelsPage} from "../pages/ModelsPage"
import {ProjectPage} from "../pages/ProjectPage"
import {ReviewPage} from "../pages/ReviewPage"
import {TrayMapPage} from "../pages/TrayMapPage"


export function App() {
  return <BrowserRouter><AppShell><Routes><Route path="/" element={<DashboardPage/>}/><Route path="/inspections" element={<InspectionPage/>}/><Route path="/trays" element={<TrayMapPage/>}/><Route path="/reviews" element={<ReviewPage/>}/><Route path="/defects" element={<DefectsPage/>}/><Route path="/alerts" element={<AlertsPage/>}/><Route path="/models" element={<ModelsPage/>}/><Route path="/project" element={<ProjectPage/>}/></Routes></AppShell></BrowserRouter>
}
