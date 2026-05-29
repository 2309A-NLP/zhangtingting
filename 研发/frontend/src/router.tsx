import {
  Navigate,
  Outlet,
  createBrowserRouter,
} from "react-router-dom";

import { AppShell } from "./ui/AppShell";
import { useAuthStore } from "./stores/auth";
import { AuthPage } from "./views/AuthPage";
import { ChatPage } from "./views/ChatPage";
import { KnowledgePage } from "./views/KnowledgePage";
import { RolesPage } from "./views/RolesPage";

function ProtectedLayout() {
  const token = useAuthStore((state) => state.token);
  if (!token) {
    return <Navigate to="/auth" replace />;
  }

  return (
    <AppShell>
      <Outlet />
    </AppShell>
  );
}

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Navigate to="/app/roles" replace />,
  },
  {
    path: "/auth",
    element: <AuthPage />,
  },
  {
    path: "/app",
    element: <ProtectedLayout />,
    children: [
      {
        index: true,
        element: <Navigate to="/app/roles" replace />,
      },
      {
        path: "roles",
        element: <RolesPage />,
      },
      {
        path: "chat",
        element: <ChatPage />,
      },
      {
        path: "knowledge",
        element: <KnowledgePage />,
      },
    ],
  },
]);
