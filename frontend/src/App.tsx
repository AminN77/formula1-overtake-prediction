import { Route, Routes } from "react-router-dom";
import { PageLayout } from "./components/layout/PageLayout";
import { BattlePageProvider } from "./context/BattlePageContext";
import { SingleBattle } from "./pages/SingleBattle";
import { BatchScoring } from "./pages/BatchScoring";
import { ModelInfo } from "./pages/ModelInfo";

export default function App() {
  return (
    <BattlePageProvider>
      <PageLayout>
        <Routes>
          <Route path="/" element={<SingleBattle />} />
          <Route path="/batch" element={<BatchScoring />} />
          <Route path="/model" element={<ModelInfo />} />
        </Routes>
      </PageLayout>
    </BattlePageProvider>
  );
}
