import { useEffect, useRef } from "react";

export default function AIAssistant({
  selectedPatient,
  userContext = {},
  ROM,
  MQ,
  COMP,
  percentage,
  llmSummary,
  head,
  left,
  right,
  keyFindings,
  figure,
  detailed_analysis,
  counterfactual_analysis,
  recommendations
}) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !selectedPatient) return;

    // Load the Invent AI script once
    if (!document.getElementById("useinvent-script")) {
      const script = document.createElement("script");
      script.src = "https://www.useinvent.com/button.js";
      script.async = true;
      script.defer = true;
      script.id = "useinvent-script";
      document.body.appendChild(script);
    }

    const timeout = setTimeout(() => {
      if (!containerRef.current) return;

      const assistant = document.createElement("invent-assistant");
      assistant.setAttribute("assistant-id", "ast_5k8YHb9LcIqNTWBR2MJ0sY");

      // Build context
      const context = {
        patient: selectedPatient,
        userContext: {
          clinical_role: userContext.clinical_role || "clinician",
          expertise_level: userContext.expertise_level || "moderate",
          dashboard_layer: userContext.dashboard_layer || "scan",
          use_case: userContext.use_case || "general",
        },
        metrics: {
          ROM,
          MQ,
          compensation: COMP,
          injured_region: { head, left, right }
        },
        probabilityInjured: percentage,
        llmSummary: {
          one_sentence_summary: llmSummary,
          key_findings: keyFindings,
          detailed_analysis,
          counterfactual_analysis,
          recommendations
        },
        trajectory3D: figure
      };

      assistant.setAttribute("data-context", JSON.stringify(context));

      // Style and position
      assistant.style.width = "50%";
      assistant.style.height = "50%";
      assistant.style.display = "block";
      assistant.style.position = "fixed";
      assistant.style.bottom = "20px";
      assistant.style.left = "20px";
      assistant.style.zIndex = "9999";

      containerRef.current.appendChild(assistant);
    }, 100);

    return () => {
      clearTimeout(timeout);
      if (containerRef.current) containerRef.current.innerHTML = "";
    };
  }, [
    selectedPatient,
    userContext,
    ROM,
    MQ,
    COMP,
    percentage,
    llmSummary,
    head,
    left,
    right,
    keyFindings,
    figure,
    detailed_analysis,
    counterfactual_analysis,
    recommendations
  ]);

  return (
    <div
      ref={containerRef}
      className="flex-1 flex flex-col p-4 min-h-0 border rounded-lg"
      style={{ height: "600px", width: "100%" }}
    >
      {!ROM && (
        <div className="flex items-center justify-center h-full text-gray-500 text-sm">
          ⚠️ Please select a patient to enable the AI assistant
        </div>
      )}
    </div>
  );
}
