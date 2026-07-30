# Three-column control-path latency figure design

## Goal

Extend the existing two-column control-path latency distribution into a
three-column publication figure that separates sender-side dispatch, Gloo
signal transfer, and receiver-side dispatch overhead.

## Data and metric definitions

All three columns use the same 56 remote communication-dependency samples from
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

The expected medians from the archived trace are approximately 183 us, 225 us,
and 346 us, respectively.

## Visual design

- Preserve the existing boxplot-plus-jittered-points visual language.
- Use three colorblind-distinguishable colors: sender blue/teal, Gloo purple,
  and receiver orange.
- Label each median directly above its distribution.
- Use short two-line x-axis labels:
  `sender complete->send`, `Gloo ready->recv`, and
  `receiver recv->submit`.
- Retain the title `Control-path latency`, y-axis in microseconds, and the
  `n = 56 dependencies; iterations 3-10` annotation.
- The sender distribution contains one approximately 1.19 ms tail sample.
  Keep the main axis comparable to the existing 0-600 us figure and mark the
  clipped tail explicitly as `1 sample at 1.19 ms`; retain the exact value in
  source data.

## Outputs

- Update the existing plotting source and regenerate the complete publication
  figure.
- Export a standalone three-column panel as editable SVG, vector PDF, and
  300-DPI PNG.
- Export auditable CSV source data with iteration, dependency ID, all three
  latency values, and units.
- Update the result README and figure caption with exact timing definitions.

## Verification

- Assert exactly 56 matched samples and identical `(iteration, dependency_id)`
  keys across all three metrics.
- Assert timestamp ordering:
  `comm_complete <= signal_send_start <= signal_recv <= target_submit`.
- Check medians against values independently calculated from the archived raw
  trace.
- Parse the generated CSV, open the PNG for visual review, and confirm the PDF
  and SVG are non-empty.
