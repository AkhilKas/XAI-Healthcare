import React, { useState } from 'react';
import { Download, User } from 'lucide-react';
import { Sidebar } from './components/Sidebar';
import { AIAssistant } from './components/AIAssistant';
import { RiskIndicator } from './components/RiskIndicator';
import { BodyDiagram } from './components/BodyDiagram';
import { Motion3D } from './components/Motion3D';
import { FindingsCard } from './components/FindingsCard';
import { AnalysisPanel } from './components/AnalysisPanel';

function App() {
  const [selectedPatient, setSelectedPatient] = useState('Sample_patient');
  const [selectedTask, setSelectedTask] = useState('Jar opening');
  const [activeView, setActiveView] = useState('overview');

  const handleAnalyze = () => setActiveView('analysis');

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar with AI Assistant */}
      <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
        <Sidebar
          selectedPatient={selectedPatient}
          setSelectedPatient={setSelectedPatient}
          selectedTask={selectedTask}
          setSelectedTask={setSelectedTask}
          onAnalyze={handleAnalyze}
        />
        <AIAssistant />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Nav */}
        <div className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setActiveView('overview')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                activeView === 'overview' ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              Glance
            </button>
            <button
              onClick={() => setActiveView('analysis')}
              className={`px-3 py-1.5 rounded-lg text-sm ${
                activeView === 'analysis' ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              Scan
            </button>
            <button
              onClick={() => setActiveView('investigate')}
              className={`px-3 py-1.5 rounded-lg text-sm ${
                activeView === 'investigate' ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              Investigate
            </button>
          </div>
          <div className="flex items-center gap-4">
            <button className="text-gray-600 hover:text-gray-800"><Download size={18} /></button>
            <button className="text-gray-600 hover:text-gray-800"><User size={18} /></button>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-auto p-6">
          {activeView === 'overview' && (
            <div className="max-w-7xl mx-auto space-y-6">
              <div className="bg-white rounded-lg shadow-sm border p-6">
                <h2 className="text-lg font-semibold mb-4">Risk Assessment Overview</h2>
                <div className="grid grid-cols-3 gap-6">
                  <RiskIndicator label="Str. Ext" value={87} color="#fbbf24" />
                  <RiskIndicator label="Scap. Subs" value={92} color="#22c55e" />
                  <RiskIndicator label="Pec Maj" value={74} color="#ef4444" />
                </div>
              </div>
            </div>
          )}

          {activeView === 'analysis' && (
            <div className="max-w-7xl mx-auto space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-white rounded-lg shadow-sm border p-6">
                  <h2 className="text-lg font-semibold mb-4">Movement Analysis</h2>
                  <BodyDiagram />
                </div>
                <div className="bg-white rounded-lg shadow-sm border p-6">
                  <h2 className="text-lg font-semibold mb-4">Key Findings</h2>
                  <div className="space-y-3">
                    <FindingsCard
                      type="error"
                      title="Abnormal Scapular Rhythm"
                      description="Elevated scapular winging detected during forward flexion"
                    />
                    <FindingsCard
                      type="warning"
                      title="Trunk Compensation Detected"
                      description="Moderate trunk flexion during reaching movements"
                    />
                    <FindingsCard
                      type="success"
                      title="Core Stability Maintained"
                      description="Good posture control throughout movement patterns"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeView === 'investigate' && (
            <div className="max-w-7xl mx-auto space-y-6">
              <div className="grid grid-cols-3 gap-6">
                <div className="col-span-2 bg-white rounded-lg shadow-sm border p-6">
                  <h2 className="text-lg font-semibold mb-4">3D Movement Reconstruction</h2>
                  <Motion3D />
                </div>
                <div className="space-y-4">
                  <AnalysisPanel
                    title="Detailed Analysis (9.3)"
                    color="blue"
                    items={[
                      'Limited ROM: 117° elevation vs normal 165°',
                      'Scapular positioning abnormal throughout arc',
                      'Compensatory trunk extension present'
                    ]}
                  />
                  <AnalysisPanel
                    title="Counterfactual Analysis (9.3)"
                    color="green"
                    items={[
                      'If scapular dyskinesis reduced by 15°',
                      'With maintained trunk posture',
                      'Classification: Improved to healthy range'
                    ]}
                  />
                  <AnalysisPanel
                    title="Recommendations"
                    color="amber"
                    items={[
                      'Focus on glenohumeral mobilization',
                      'Strengthen scapular stabilizers',
                      'Address postural compensation'
                    ]}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;