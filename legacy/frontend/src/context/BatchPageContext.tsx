import { createContext, useContext, useMemo, useState, type Dispatch, type ReactNode, type SetStateAction } from "react";
import type { BatchFilterOptions, BatchPredictResponse } from "../types";

export type BatchViewerFilters = {
  outcome: string;
  prediction: string;
  attacker: string;
  defender: string;
  race_name: string;
  track: string;
  search: string;
  lap_min: string;
  lap_max: string;
  probability_min: string;
};

type BatchPageContextValue = {
  file: File | null;
  setFile: Dispatch<SetStateAction<File | null>>;
  threshold: number;
  setThreshold: Dispatch<SetStateAction<number>>;
  filterPits: boolean;
  setFilterPits: Dispatch<SetStateAction<boolean>>;
  loading: boolean;
  setLoading: Dispatch<SetStateAction<boolean>>;
  err: string | null;
  setErr: Dispatch<SetStateAction<string | null>>;
  result: BatchPredictResponse | null;
  setResult: Dispatch<SetStateAction<BatchPredictResponse | null>>;
  filterOptions: BatchFilterOptions;
  setFilterOptions: Dispatch<SetStateAction<BatchFilterOptions>>;
  page: number;
  setPage: Dispatch<SetStateAction<number>>;
  pageSize: number;
  setPageSize: Dispatch<SetStateAction<number>>;
  filters: BatchViewerFilters;
  setFilters: Dispatch<SetStateAction<BatchViewerFilters>>;
  selectedRow: Record<string, unknown> | null;
  setSelectedRow: Dispatch<SetStateAction<Record<string, unknown> | null>>;
  modalOpen: boolean;
  setModalOpen: Dispatch<SetStateAction<boolean>>;
  resetViewerState: () => void;
};

const DEFAULT_FILTERS: BatchViewerFilters = {
  outcome: "ALL",
  prediction: "ALL",
  attacker: "ALL",
  defender: "ALL",
  race_name: "ALL",
  track: "ALL",
  search: "",
  lap_min: "",
  lap_max: "",
  probability_min: "",
};

const BatchPageContext = createContext<BatchPageContextValue | null>(null);

export function BatchPageProvider({ children }: { children: ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [filterPits, setFilterPits] = useState(true);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<BatchPredictResponse | null>(null);
  const [filterOptions, setFilterOptions] = useState<BatchFilterOptions>({});
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [filters, setFilters] = useState<BatchViewerFilters>(DEFAULT_FILTERS);
  const [selectedRow, setSelectedRow] = useState<Record<string, unknown> | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const resetViewerState = () => {
    setFilters(DEFAULT_FILTERS);
    setPage(1);
    setPageSize(25);
    setSelectedRow(null);
    setModalOpen(false);
  };

  const value = useMemo(
    () => ({
      file,
      setFile,
      threshold,
      setThreshold,
      filterPits,
      setFilterPits,
      loading,
      setLoading,
      err,
      setErr,
      result,
      setResult,
      filterOptions,
      setFilterOptions,
      page,
      setPage,
      pageSize,
      setPageSize,
      filters,
      setFilters,
      selectedRow,
      setSelectedRow,
      modalOpen,
      setModalOpen,
      resetViewerState,
    }),
    [
      err,
      file,
      filterOptions,
      filterPits,
      filters,
      loading,
      modalOpen,
      page,
      pageSize,
      result,
      selectedRow,
      threshold,
    ],
  );

  return <BatchPageContext.Provider value={value}>{children}</BatchPageContext.Provider>;
}

export function useBatchPageState(): BatchPageContextValue {
  const ctx = useContext(BatchPageContext);
  if (!ctx) throw new Error("useBatchPageState must be used within BatchPageProvider");
  return ctx;
}
