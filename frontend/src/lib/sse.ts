/**
 * Read a Server-Sent Events body and yield the text after each "data: ".
 *
 * Network chunks do not line up with events: one event can arrive split across two
 * chunks. So the text is buffered and only complete events (ended by a blank line)
 * are handed on.
 */
export async function* readSse(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end: number;
      while ((end = buffer.indexOf("\n\n")) !== -1) {
        const event = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        for (const line of event.split("\n")) {
          if (line.startsWith("data: ")) yield line.slice(6);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
