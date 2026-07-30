# Three-column control-path latency figure design

## Goal

Extend the existing two-column control-path latency distribution into a
three-column publication figure that separates sender-side dispatch, Gloo
signal transfer, and receiver-side dispatch overhead.

## Data and metric definitions

The source trace contains 56 remote communication-dependency samples from
formal iterations 3-10 of the existing `C_TRACE` run.

1. **Sender dispatch** (`complete -> send`):
   `signal_send_start_ns - trigger_comm_complete_ns`.
2. **Signal transfer** (`ready -> recv`):
   `signal_recv_ns - max(signal_send_start_ns, signal_wait_start_ns)`.
   This retains the existing Gloo metric and excludes time before either Gloo
   endpoint has entered its operation.
3. **Receiver dispatch** (`recv -> submit`):
   `target_submit_ns - signal_recv_ns`.
   This is the existing orange distribution, moved unchanged from column two
   to column three.

The sender-side plot excludes the single approximately 1.19 ms tail sample,
leaving 55 displayed sender samples with a median of approximately 183 us.
The Gloo and receiver columns retain all 56 samples, with medians of
approximately 225 us and 346 us, respectively.

## Visual design

- Preserve the existing boxplot-plus-jittered-points visual language.
- Use three colorblind-distinguishable colors: sender blue/teal, Gloo purple,
  and receiver orange.
- Label each median directly above its distribution.
- Use short two-line x-axis labels:
  `sender complete->send`, `Gloo ready->recv`, and
  `receiver recv->submit`.
- Retain the title `Control-path latency`, y-axis in microseconds, and the
  iterations 3-10 annotation.
- Keep the main axis comparable to the existing 0-600 us figure. Do not draw
  or annotate the approximately 1.19 ms sender-side tail sample. State the
  displayed sample counts as `sender n = 55; Gloo/receiver n = 56`.
- Preserve the excluded raw sample in the auditable source CSV with an
  `included_in_plot=false` field.

## Outputs

- Update the existing plotting source and regenerate the complete publication
  figure.
- Export a standalone three-column panel as editable SVG, vector PDF, and
  300-DPI PNG.
- Export auditable CSV source data with iteration, dependency ID, all three
  latency values, and units.
- Update the result README and figure caption with exact timing definitions.

## Verification

- Assert exactly 56 matched raw samples and identical
  `(iteration, dependency_id)` keys across all three metrics.
- Assert exactly one sender sample is excluded by the documented 600 us
  plotting cutoff, leaving 55 displayed sender samples.
- Assert timestamp ordering:
  `comm_complete <= signal_send_start <= signal_recv <= target_submit`.
- Check medians against values independently calculated from the archived raw
  trace.
- Parse the generated CSV, open the PNG for visual review, and confirm the PDF
  and SVG are non-empty.
