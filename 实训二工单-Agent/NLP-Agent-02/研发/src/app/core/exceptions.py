class ApplicationError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = "APPLICATION_ERROR",
        status_code: int = 400,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(ApplicationError):
    def __init__(self, message: str, *, error_code: str = "NOT_FOUND") -> None:
        super().__init__(message, error_code=error_code, status_code=404)


class UnauthorizedError(ApplicationError):
    def __init__(self, message: str, *, error_code: str = "UNAUTHORIZED") -> None:
        super().__init__(message, error_code=error_code, status_code=401)
