import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, Mock, patch

import aiohttp

from github_stats import Queries


class QueryRestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = Mock()
        self.session.get = AsyncMock()
        self.queries = Queries("octocat", "test-token", self.session)

        sleep_patch = patch("github_stats.asyncio.sleep", new_callable=AsyncMock)
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

        fallback_patch = patch("github_stats.requests.get")
        self.fallback = fallback_patch.start()
        self.fallback.return_value.status_code = 599
        self.addCleanup(fallback_patch.stop)

    @staticmethod
    def response(status, data=None):
        response = Mock()
        response.status = status
        response.json = AsyncMock(return_value=data)
        return response

    async def call(self, path, params=None):
        output = io.StringIO()
        with redirect_stdout(output):
            result = await self.queries.query_rest(path, params)
        return result, output.getvalue()

    async def test_no_content_contributors_is_empty_without_retry_or_json(self):
        response = self.response(204)
        self.session.get.return_value = response

        result, output = await self.call("/repos/octocat/project/stats/contributors")

        self.assertEqual(result, [])
        self.session.get.assert_awaited_once()
        response.json.assert_not_awaited()
        self.fallback.assert_not_called()
        self.sleep.assert_not_awaited()
        self.assertNotIn("incomplete", output.lower())

    async def test_no_content_traffic_is_empty_mapping(self):
        response = self.response(204)
        self.session.get.return_value = response

        result, _ = await self.call("/repos/octocat/project/traffic/views")

        self.assertEqual(result, {})
        self.session.get.assert_awaited_once()
        response.json.assert_not_awaited()
        self.sleep.assert_not_awaited()
        self.fallback.assert_not_called()

    async def test_pending_statistics_are_retried_until_ready(self):
        pending = self.response(202)
        data = [{"author": {"login": "octocat"}, "weeks": [{"a": 12, "d": 3}]}]
        ready = self.response(200, data)
        self.session.get.side_effect = [pending, ready]

        result, _ = await self.call("/repos/octocat/project/stats/contributors")

        self.assertEqual(result, data)
        self.assertEqual(self.session.get.await_count, 2)
        self.sleep.assert_awaited_once()
        pending.json.assert_not_awaited()
        self.fallback.assert_not_called()

    async def test_pending_then_no_content_stops_without_false_incomplete_warning(self):
        pending = self.response(202)
        empty = self.response(204)
        self.session.get.side_effect = [pending, empty]

        result, output = await self.call("/repos/octocat/project/stats/contributors")

        self.assertEqual(result, [])
        self.assertEqual(self.session.get.await_count, 2)
        self.sleep.assert_awaited_once()
        empty.json.assert_not_awaited()
        self.fallback.assert_not_called()
        self.assertNotIn("incomplete", output.lower())

    async def test_terminal_http_errors_are_reported_without_returning_error_json(self):
        for status in (403, 404, 422):
            with self.subTest(status=status):
                self.session.get.reset_mock()
                response = self.response(status, {"message": "Request rejected"})
                self.session.get.return_value = response

                result, output = await self.call("/repos/octocat/project/stats/contributors")

                self.assertEqual(result, {})
                self.assertIn(f"HTTP {status}", output)
                self.assertNotIn("too many 202", output.lower())
                self.session.get.assert_awaited_once()
                response.json.assert_not_awaited()
                self.fallback.assert_not_called()
                self.sleep.assert_not_awaited()

    async def test_pending_statistics_have_a_bounded_retry_budget(self):
        response = self.response(202)
        self.session.get.return_value = response

        result, output = await self.call("/repos/octocat/project/stats/contributors")

        self.assertEqual(result, {})
        self.assertEqual(self.session.get.await_count, 60)
        self.assertIn("HTTP 202", output)
        self.assertIn("incomplete", output.lower())
        response.json.assert_not_awaited()
        self.fallback.assert_not_called()

    async def test_fallback_no_content_is_terminal_without_json_or_retry(self):
        self.session.get.side_effect = aiohttp.ClientConnectionError("Connection lost")
        response = Mock(status_code=204)
        self.fallback.return_value = response

        result, output = await self.call("/repos/octocat/project/stats/contributors")

        self.assertEqual(result, [])
        self.session.get.assert_awaited_once()
        self.fallback.assert_called_once()
        response.json.assert_not_called()
        self.sleep.assert_not_awaited()
        self.assertNotIn("incomplete", output.lower())

    async def test_public_repository_can_be_identified_in_error_logs(self):
        self.queries.public_repos.add("octocat/public-project")
        self.session.get.return_value = self.response(403)

        _, output = await self.call("/repos/octocat/public-project/stats/contributors")

        self.assertIn("octocat/public-project", output)
        self.assertIn("HTTP 403", output)
        self.assertNotIn("test-token", output)

    async def test_unknown_repository_and_exception_url_are_not_exposed(self):
        path = "/repos/octocat/confidential-project/stats/contributors"
        self.session.get.side_effect = aiohttp.ClientConnectionError(
            "Failed https://api.github.com" + path + "?access_token=test-token"
        )
        self.fallback.return_value.status_code = 403

        result, output = await self.call(path)

        self.assertEqual(result, {})
        self.assertIn("HTTP 403", output)
        self.assertNotIn("confidential-project", output)
        self.assertNotIn("octocat", output)
        self.assertNotIn("test-token", output)
        self.assertNotIn("https://", output)
        self.session.get.assert_awaited_once()
        self.fallback.assert_called_once()
        self.sleep.assert_not_awaited()

    async def test_success_preserves_response_and_query_parameters(self):
        data = {"views": [{"count": 5}]}
        self.session.get.return_value = self.response(200, data)

        result, _ = await self.call(
            "/repos/octocat/project/traffic/views", {"per": "week"}
        )

        self.assertEqual(result, data)
        self.assertEqual(
            self.session.get.call_args.kwargs["params"], (("per", "week"),)
        )
        self.session.get.assert_awaited_once()
        self.fallback.assert_not_called()
        self.sleep.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
