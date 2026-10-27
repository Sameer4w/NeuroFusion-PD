
import gradio as gr

with gr.Blocks(title="NeuroFusion-PD") as demo:
    gr.Markdown("# 🧠 NeuroFusion-PD")
    gr.Markdown(
        "Multi-modal Parkinson's disease screening using "
        "Voice, Handwriting, and Gait."
    )

    modality = gr.CheckboxGroup(
        choices=["Voice", "Handwriting", "Gait"],
        label="Select available modalities",
        info="Select one or more modalities."
    )

    fusion_method = gr.Radio(
        choices=[
            "Average",
            "Maximum Probability",
            "Reliability Weighted"
        ],
        value="Average",
        label="Fusion Method"
    )

    voice_input = gr.Audio(
        label="Voice Recording",
        type="filepath",
        format="wav"
    )

    handwriting_input = gr.File(
        label="Handwriting Trajectory (.txt)",
        file_types=[".txt"],
        type="filepath"
    )

    gait_input = gr.File(
        label="Gait Sensor Recording (.zip)",
        file_types=[".zip"],
        type="filepath"
    )

    predict_button = gr.Button(
        "🔍 Analyze for Parkinson's",
        variant="primary"
    )

    result_output = gr.JSON(
        label="Prediction Results"
    )

    predict_button.click(
        fn=analyze_patient,
        inputs=[
            modality,
            fusion_method,
            voice_input,
            handwriting_input,
            gait_input
        ],
        outputs=result_output
    )

demo.launch(share=True, debug=True)
