import type { PerceptualParams } from "@/lib/types/ui";

type ParamListProps = {
  params: PerceptualParams;
};

export function ParamList({ params }: ParamListProps) {
  return (
    <div className="grid grid-cols-2 gap-3 text-sm">
      {Object.entries(params).map(([key, value]) => (
        <div key={key} className="rounded-2xl border border-white/10 bg-white/5 p-3">
          <div className="text-xs uppercase tracking-wide text-white/45">{key}</div>
          <div className="mt-1 font-medium text-white">{value.toFixed(2)}</div>
        </div>
      ))}
    </div>
  );
}
