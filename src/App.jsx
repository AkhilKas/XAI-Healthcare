import React, { useState } from 'react';
import { Download, User } from 'lucide-react';
import { Sidebar } from './components/Sidebar';
import { AIAssistant } from './components/AIAssistant';
import { RiskIndicator } from './components/RiskIndicator';
import { BodyDiagram } from './components/BodyDiagram';
import { FindingsCard } from './components/FindingsCard';
import { AnalysisPanel } from './components/AnalysisPanel';
import { InjuryPredictionCard } from './components/InjuryPredictionCard';
import { FigureDisplay } from './components/FigureDisplay';

function App() {
  const [selectedPatient, setSelectedPatient] = useState('patient_1');
  const [selectedTask, setSelectedTask] = useState('Jar opening');
  const [activeView, setActiveView] = useState('overview');

  const [loading, setLoading] = useState(false);
  const [disabled, setDisabled] = useState(true);
  // const [analysisData, setAnalysisData] = useState(null);
  const [ROM, setROM] = useState(33);
  const [MQ, setMQ] = useState(66);
  const [COMP, setComp] = useState(100);

  const [percentage, setPercentage] = useState(0);
  const [llmSummary, setLlmSummary] = useState("Loading overview...");

  const [head, setHead] = useState(0);
  const [left, setLeft] = useState(0);
  const [right, setRight] = useState(0);

  const initialFindings = {
    "Error: Finding 1": "key finding 1 description",
    "Warning: Finding 2": "key finding 2 description",
    "Success: Finding 3": "key finding 3 description",
  };
  const [keyFindings, setKeyFindings] = useState(initialFindings);
  const [figure, setFigure] = useState("Loading figure...");

  const [detailed_analysis, setDetailedAnalysis] = useState([])
  const [counterfactual_analysis, setCounterfactualAnalysis] = useState([])
  const [recommendations, setRecommendations] = useState([])

  const handleGo = async () => {
    setActiveView('overview');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/go', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          patient: selectedPatient,
          task: selectedTask
        })
      });

      const data = await response.json();
      console.log("API Response:", data);

      setDisabled(false);

      setROM(parseInt(data.metrics.aggregated_rom))
      setMQ(parseInt(data.metrics.aggregated_mq))
      setComp(parseInt(data.metrics.compensation))

      setPercentage(parseInt(data.probability_injured * 100))
      setLlmSummary(data.llm_summary.one_sentence_summary);

      setHead(data.metrics.injured_region.head)
      setLeft(data.metrics.injured_region.left)
      setRight(data.metrics.injured_region.right)

      setKeyFindings(data.llm_summary.key_findings);
      setFigure(data.trajectory_3d);

      setDetailedAnalysis(data.llm_summary.detailed_analysis);
      setCounterfactualAnalysis(data.llm_summary.counterfactual_analysis);
      setRecommendations(data.llm_summary.recommendations);

    } catch (error) {
      console.error("Error while analyzing:", error);
    }
    finally {
      setLoading(false); // stop loader
    }
  };


  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar with AI Assistant */}
      <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
        <Sidebar
          selectedPatient={selectedPatient}
          setSelectedPatient={setSelectedPatient}
          selectedTask={selectedTask}
          setSelectedTask={setSelectedTask}
          onAnalyze={handleGo}
          loading={loading}
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
                <h2 className="text-lg font-semibold mb-4">Risk Assessment</h2>
                <div className="grid grid-cols-3 gap-6">
                  <RiskIndicator label="Range of Motion" value={ROM} color={ROM <= 33 ? "#ef4444" : ROM <= 66 ? "#fbbf24" : "#22c55e"} style={{ opacity: disabled ? 0.5 : 1 }} />
                  <RiskIndicator label="Movement Quality" value={MQ} color={MQ <= 33 ? "#ef4444" : MQ <= 66 ? "#fbbf24" : "#22c55e"}  style={{ opacity: disabled ? 0.5 : 1 }} />
                  <RiskIndicator label="Head Compensation" value={COMP} color={COMP <= 33 ? "#ef4444" : COMP <= 66 ? "#fbbf24" : "#22c55e"} style={{ opacity: disabled ? 0.5 : 1 }} />
                </div>
              </div>
              <div className="bg-white rounded-lg shadow-sm border p-6">
                <div className="grid grid-cols-2 gap-6">
                  < InjuryPredictionCard label="Injury Prediction" status={percentage > 50 ? "Injured" : "Non-Injured"} percentage={percentage} opacity={disabled ? 0.5 : 1} />
                  <div className="bg-white rounded-lg shadow-sm border p-6" >
                    <h2 className="text-lg font-semibold mb-4">Overview</h2>
                    <p className="text-sm text-gray-700 mb-4" style={{ opacity: disabled ? 0.5 : 1 }}>{llmSummary}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeView === 'analysis' && (
            <div className="max-w-7xl mx-auto space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-white rounded-lg shadow-sm border p-6">
                  <h2 className="text-lg font-semibold mb-4">Movement Analysis</h2>
                  <BodyDiagram values={{ head, left, right }} style={{ opacity: disabled ? 0.5 : 1 }} />
                </div>
                <div className="bg-white rounded-lg shadow-sm border p-6">
                  <h2 className="text-lg font-semibold mb-4">Key Findings</h2>
                  <div className="space-y-3" style={{ opacity: disabled ? 0.5 : 1 }}>
                    {Object.entries(keyFindings).map(([title, description], index) => (
                      <FindingsCard
                        key={index}
                        type={title.toLowerCase().includes("error")
                          ? "error"
                          : title.toLowerCase().includes("warning")
                          ? "warning"
                          : "success"}
                        title={title}
                        description={description}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeView === 'investigate' && (
            <div className="max-w-7xl mx-auto space-y-6">
              <div className="grid grid-cols-3 gap-6">
                <div className="col-span-2 bg-white rounded-lg shadow-sm border p-6">
                  <h2 className="text-lg font-semibold mb-4">3D Plot</h2>
                  <FigureDisplay imgBase64={figure} style={{ opacity: disabled ? 0.5 : 1 }}/>
                </div>
                <div className="space-y-4">
                  <AnalysisPanel
                    title="Detailed Analysis"
                    color="blue"
                    items={detailed_analysis}
                  />
                  <AnalysisPanel
                    title="Counterfactual Analysis"
                    color="green"
                    items={counterfactual_analysis}
                  />
                  <AnalysisPanel
                    title="Recommendations"
                    color="amber"
                    items={recommendations}
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