Patch v20: desktop-like audio bridge lifecycle

Replace these files in research/earloop_personalization_cli:
- earloop_cli/cli.py
- earloop_cli/live_session.py
- earloop_cli/audio_bridge.py
- earloop_cli/config.py
- config.example.json

Why:
- Desktop version keeps the audio bridge alive and does not recreate PortAudio/MME streams between personalization sessions.
- CLI previously recreated/stopped the bridge for every session. On some Windows + MME + VB-Cable setups this can start in silence when Windows Output is already CABLE Input.

What changed:
- One audio bridge instance is created for the whole CLI run.
- New personalization sessions reuse the existing bridge.
- LiveSession no longer stops the bridge after every session when keep_bridge_alive_between_sessions=true.
- The bridge is stopped only when the CLI exits.
- RealtimeAudioBridge.start/stop are idempotent and expose running state.
- Added config: audio.keep_bridge_alive_between_sessions=true.

Recommended route:
1. Before launching CLI bridge: Windows Output = headphones/speakers.
2. CLI starts bridge with locked capture/playback route.
3. After [audio] bridge started: Windows Output = CABLE Input.
4. If you start a new session inside CLI, do not switch Windows Output back; the existing bridge is reused.
