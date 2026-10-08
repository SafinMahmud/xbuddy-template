import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Model output rendered as Markdown. GFM adds tables (the roadmap's weekly plan)
 * and task lists. Links, such as job postings from a search, open in a new tab.
 */
export default function Markdown({ children }: { children: string }) {
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children: label }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {label}
            </a>
          ),
          table: ({ children: rows }) => (
            <div className="table-scroll">
              <table>{rows}</table>
            </div>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
