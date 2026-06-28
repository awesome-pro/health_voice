"use client";

/* -------------------------------------------------------------------------- */
/*  UI primitives — a small, shadcn-style component kit (Tailwind, no Radix).  */
/*  Shared across the HealthVoice console so every surface uses the same       */
/*  cards, buttons, badges, inputs, and headers.                               */
/* -------------------------------------------------------------------------- */
import { cva, type VariantProps } from "class-variance-authority";
import {
  type ButtonHTMLAttributes,
  type HTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  forwardRef,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { Info, Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";

/* ----------------------------------- Card --------------------------------- */

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-clinical-border bg-clinical-panel shadow-card",
        className
      )}
      {...props}
    />
  );
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-5 sm:p-6", className)} {...props} />;
}

/* -------------------------------- InfoHint -------------------------------- */

/**
 * A small info icon that reveals a hover card with supporting detail. The card
 * is portaled to <body> with fixed positioning, so a scrolling or overflow
 * container can never clip it, and it flips below the icon when there is no
 * room above. Hover and keyboard-focus both open it.
 */
export function InfoHint({
  children,
  label = "More detail",
  className,
}: {
  children: ReactNode;
  label?: string;
  className?: string;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const [tip, setTip] = useState<{
    left: number;
    top: number;
    place: "above" | "below";
  } | null>(null);

  const open = () => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const place = r.top < 140 ? "below" : "above";
    const left = Math.max(
      150,
      Math.min(r.left + r.width / 2, window.innerWidth - 150)
    );
    setTip({ left, top: place === "above" ? r.top : r.bottom, place });
  };
  const close = () => setTip(null);

  return (
    <button
      type="button"
      ref={ref}
      aria-label={label}
      onMouseEnter={open}
      onMouseLeave={close}
      onFocus={open}
      onBlur={close}
      className={cn(
        "inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-slate-400 transition-colors hover:text-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-accent/40",
        className
      )}
    >
      <Info className="h-3.5 w-3.5" strokeWidth={2} />
      {tip &&
        createPortal(
          <span
            role="tooltip"
            style={{
              position: "fixed",
              left: tip.left,
              top: tip.top,
              maxWidth: 280,
              transform:
                tip.place === "above"
                  ? "translate(-50%, calc(-100% - 10px))"
                  : "translate(-50%, 10px)",
            }}
            className="pointer-events-none z-[60] block rounded-xl bg-slate-900 px-3.5 py-2.5 text-left text-xs font-normal leading-relaxed text-slate-100 shadow-xl ring-1 ring-black/10"
          >
            {children}
            <span
              className={cn(
                "absolute left-1/2 h-2.5 w-2.5 -translate-x-1/2 rotate-45 bg-slate-900",
                tip.place === "above"
                  ? "top-full -mt-1.5"
                  : "bottom-full -mb-1.5"
              )}
            />
          </span>,
          document.body
        )}
    </button>
  );
}

/**
 * A consistent panel header: an icon tile, a title (with optional info hover
 * card) + optional description, and an optional right-aligned actions slot.
 * Used at the top of every Card.
 */
export function SectionHeader({
  icon,
  title,
  info,
  description,
  actions,
  divided = false,
  className,
}: {
  icon?: ReactNode;
  title: ReactNode;
  info?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  divided?: boolean;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-start justify-between gap-3",
        divided && "border-b border-clinical-border px-5 py-4 sm:px-6",
        className
      )}
    >
      <div className="flex min-w-0 items-start gap-3">
        {icon && (
          <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-clinical-accentSoft text-clinical-accent">
            {icon}
          </span>
        )}
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-[15px] font-semibold tracking-tight text-slate-900">
              {title}
            </h2>
            {info && <InfoHint>{info}</InfoHint>}
          </div>
          {description && (
            <p className="mt-1 text-[13px] leading-relaxed text-slate-500">
              {description}
            </p>
          )}
        </div>
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

/* ---------------------------------- Button -------------------------------- */

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-60",
  {
    variants: {
      variant: {
        primary:
          "bg-clinical-accent text-white shadow-sm hover:bg-clinical-accentDark focus-visible:ring-clinical-accent",
        destructive:
          "bg-red-600 text-white shadow-sm hover:bg-red-700 focus-visible:ring-red-500",
        secondary:
          "border border-clinical-border bg-white text-slate-700 shadow-sm hover:bg-slate-50 hover:text-slate-900 focus-visible:ring-slate-300",
        ghost:
          "text-slate-500 hover:bg-slate-100 hover:text-slate-800 focus-visible:ring-slate-300",
        link: "text-clinical-accent hover:underline focus-visible:ring-clinical-accent",
      },
      size: {
        sm: "h-8 px-3 text-[13px]",
        md: "h-10 px-4 text-sm",
        lg: "h-12 px-6 text-base",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  }
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, loading, children, disabled, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  )
);
Button.displayName = "Button";

/* ---------------------------------- Badge --------------------------------- */

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
  {
    variants: {
      tone: {
        neutral: "border-slate-200 bg-slate-50 text-slate-600",
        accent: "border-clinical-accentSoft bg-clinical-accentSoft text-clinical-accentDark",
        success: "border-emerald-200 bg-emerald-50 text-emerald-700",
        warning: "border-amber-200 bg-amber-50 text-amber-700",
        danger: "border-red-200 bg-red-50 text-red-700",
        info: "border-sky-200 bg-sky-50 text-sky-700",
        violet: "border-violet-200 bg-violet-50 text-violet-700",
      },
    },
    defaultVariants: { tone: "neutral" },
  }
);

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}

/** A small status pill with a leading colored dot (optionally pulsing). */
export function StatusBadge({
  tone = "neutral",
  pulse = false,
  children,
  className,
}: {
  tone?: NonNullable<BadgeProps["tone"]>;
  pulse?: boolean;
  children: ReactNode;
  className?: string;
}) {
  const dot: Record<NonNullable<BadgeProps["tone"]>, string> = {
    neutral: "bg-slate-400",
    accent: "bg-clinical-accent",
    success: "bg-emerald-500",
    warning: "bg-amber-500",
    danger: "bg-red-500",
    info: "bg-sky-500",
    violet: "bg-violet-500",
  };
  return (
    <Badge tone={tone} className={className}>
      <span className={cn("h-2 w-2 rounded-full", dot[tone], pulse && "animate-pulseDot")} />
      {children}
    </Badge>
  );
}

/* ------------------------------- Input / Label ---------------------------- */

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "w-full rounded-lg border border-clinical-border bg-white px-3 py-2 text-sm text-slate-800 shadow-sm transition placeholder:text-slate-400 focus:border-clinical-accent focus:outline-none focus:ring-2 focus:ring-clinical-accent/30 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400",
        className
      )}
      {...props}
    />
  )
);
Input.displayName = "Input";

export function FieldLabel({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "text-[11px] font-semibold uppercase tracking-wide text-slate-500",
        className
      )}
      {...props}
    />
  );
}

/* ------------------------------- Misc helpers ----------------------------- */

export function Separator({ className }: { className?: string }) {
  return <div className={cn("h-px w-full bg-clinical-border", className)} />;
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("h-4 w-4 animate-spin", className)} />;
}
