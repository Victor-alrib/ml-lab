import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, Navigate, RouterProvider } from 'react-router-dom'
import '@fontsource/barlow/400.css'
import '@fontsource/barlow/500.css'
import '@fontsource/barlow/600.css'
import '@fontsource/barlow-semi-condensed/500.css'
import '@fontsource/barlow-semi-condensed/600.css'
import '@fontsource/barlow-semi-condensed/700.css'
import '@fontsource/ibm-plex-mono/400.css'
import './index.css'
import { AppProvider } from './context/AppContext'
import { Layout } from './components/Layout'
import { RouteError } from './components/RouteError'
import DatasetsPage from './pages/Datasets'
import EDAPage from './pages/EDA'
import PreprocessPage from './pages/Preprocess'
import TrainPage from './pages/Train'
import EvaluatePage from './pages/Evaluate'
import ExperimentsPage from './pages/Experiments'
import LearnPage from './pages/Learn'

const router = createBrowserRouter([
  {
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Navigate to="/datasets" replace /> },
      { path: 'datasets', element: <DatasetsPage /> },
      { path: 'eda', element: <EDAPage /> },
      { path: 'preprocess', element: <PreprocessPage /> },
      { path: 'train', element: <TrainPage /> },
      { path: 'evaluate', element: <EvaluatePage /> },
      { path: 'evaluate/:id', element: <EvaluatePage /> },
      { path: 'experiments', element: <ExperimentsPage /> },
      { path: 'learn', element: <LearnPage /> },
      { path: '*', element: <Navigate to="/datasets" replace /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppProvider>
      <RouterProvider router={router} />
    </AppProvider>
  </StrictMode>,
)
