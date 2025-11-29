import React from 'react';

export const Sidebar = ({ selectedPatient, setSelectedPatient, selectedTask, setSelectedTask, onAnalyze }) => {
  return (
    <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-800">XAI Healthcare</h1>
        <p className="text-xs text-gray-500 mt-1">Motion Assessment Platform</p>
      </div>

      <div className="p-4 space-y-4">
        <div>
          <label className="text-xs font-medium text-gray-700 block mb-2">Select Patient</label>
          <select 
            value={selectedPatient}
            onChange={(e) => setSelectedPatient(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option>Sample_patient</option>
            <option>Sample_healthy</option>
            <option>Patient_003</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-medium text-gray-700 block mb-2">Select Task</label>
          <select 
            value={selectedTask}
            onChange={(e) => setSelectedTask(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option>Jar opening</option>
            <option>Key turning</option>
            <option>Wall cleaning</option>
            <option>Backwashing</option>
            <option>Knife slicing</option>
          </select>
        </div>

        <button 
          onClick={onAnalyze}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors"
        >
          GO
        </button>
      </div>
    </div>
  );
};